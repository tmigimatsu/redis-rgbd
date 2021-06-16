/**
 * main.cc
 *
 * Copyright 2019. All Rights Reserved.
 *
 * Created: July 29, 2019
 * Authors: Toki Migimatsu
 */

#include <ctrl_utils/argparse.h>
#include <ctrl_utils/atomic.h>
#include <ctrl_utils/eigen_string.h>
#include <ctrl_utils/opencv.h>
#include <ctrl_utils/redis_client.h>
#include <ctrl_utils/semaphore.h>
#include <ctrl_utils/thread_pool.h>
#include <ctrl_utils/timer.h>
#include <redis_rgbd/kinect2.h>

#include <csignal>    // std::signal, std::sig_atomic_t
#include <exception>  // std::invalid_argument
#include <iostream>   // std::cout

namespace {

volatile std::sig_atomic_t g_runloop = true;
void Stop(int signal) { g_runloop = false; }

enum class ImageType { Color, Depth };

struct Args : ctrl_utils::Args {
  explicit Args(ctrl_utils::Args&& args) : ctrl_utils::Args(std::move(args)) {}

  virtual std::string_view description() const override {
    return "Calibrate the camera's position by clicking on end-effector "
           "positions in the image.";
  }

  std::string camera =
      Arg<std::string>("camera", "One of {kinect, kinect2, realsense}.");

  std::string serial =
      Kwarg<std::string>("serial", "", "Camera serial number.");

  std::string redis_host = Kwarg<std::string>("h,redis-host", "127.0.0.1",
                                              "Redis hostname for the robot.");

  int redis_port =
      Kwarg<int>("p,redis-port", 6379, "Redis port for the robot.");

  std::string redis_pass =
      Kwarg<std::string>("a,redis-pass", "", "Redis password for the robot.");

  std::string key_pos =
      Kwarg<std::string>("pos-key", "franka_panda::sensor::pos",
                         "Redis key for the robot's end-effector position.");

  std::string key_prefix_camera = Kwarg<std::string>(
      "key-prefix", "rgbd::camera_0::",
      "Redis key prefix for the camera's pose. The quaternion will be set to "
      "<key-prefix>ori and the position to <key-prefix>pos.");

  bool verbose = Flag("verbose", false, "Print camera logging output.");
};

struct CameraRobotState {
  // State updated by main loop.
  cv::Point3d point_ee;
  cv::Mat img_color;
  cv::Mat img_depth;
  std::binary_semaphore ownership = std::binary_semaphore(false);

  // Queue of mouse clicks.
  std::vector<cv::Point3d> points_ee;
  std::vector<cv::Point2d> points_img;
  std::binary_semaphore data_ready = std::binary_semaphore(false);
};

}  // namespace

int main(int argc, char* argv[]) {
  std::signal(SIGINT, Stop);

  std::optional<Args> args = ctrl_utils::ParseArgs<Args>(argc, argv);
  if (!args.has_value()) return 1;
  std::cout << args->help_string() << std::endl << *args << std::endl;

  // Connect to camera.
  std::cout << "Connecting to " << args->camera;
  if (!args->serial.empty()) {
    std::cout << " with serial " << args->serial;
  }
  std::cout << "... " << std::endl;

  std::shared_ptr<redis_rgbd::Camera> camera;
  if (args->camera == "kinect2") {
    camera = std::make_shared<redis_rgbd::Kinect2>(args->verbose);
  } else {
    std::cerr << args->camera << " is not supported." << std::endl;
    return 1;
  }
  const auto* kinect2 = dynamic_cast<redis_rgbd::Kinect2*>(camera.get());

  const bool is_connected = camera->Connect(args->serial);
  if (!is_connected) return 1;
  std::cout << "Done." << std::endl;

  // Connect to Redis.
  std::cout << "Connecting to robot Redis server at " << args->redis_host << ":"
            << args->redis_port << "... " << std::flush;

  ctrl_utils::RedisClient redis;
  redis.connect(args->redis_host, args->redis_port, args->redis_pass);
  std::cout << "Done." << std::endl;

  // Start listening.
  camera->Start(true, true);

  // Preallocate registered images.
  CameraRobotState state;
  state.img_color = cv::Mat(camera->color_height(), camera->color_width(),
                            camera->color_channel());
  state.img_depth = cv::Mat(camera->color_height(), camera->color_width(),
                            camera->depth_channel());
  cv::Mat img_depth_registered;

  // Start calibration thread.
  std::thread thread_calibrate([&args, camera, &redis, &state]() {
    const std::string KEY_POS = args->key_prefix_camera + "pos";
    const std::string KEY_ORI = args->key_prefix_camera + "ori";

    std::vector<cv::Point3d> points_ee;
    std::vector<cv::Point2d> points_img;

    while (g_runloop) {
      // Get data when available.
      state.data_ready.acquire();
      if (!g_runloop) break;

      points_ee.insert(points_ee.end(),
                       std::make_move_iterator(state.points_ee.begin()),
                       std::make_move_iterator(state.points_ee.end()));
      state.points_ee.clear();
      points_img.insert(points_img.end(),
                        std::make_move_iterator(state.points_img.begin()),
                        std::make_move_iterator(state.points_img.end()));
      state.points_img.clear();
      state.ownership.release();

      std::cout << std::endl
                << "Click: " << points_img.back() << std::endl
                << "End-effector position: " << points_ee.back() << std::endl;

      if (points_ee.size() < 4) continue;

      // Optimize.
      std::cout << std::endl << "Optimizing... " << std::flush;
      cv::Vec3d rvec, tvec;
      cv::solvePnP(points_ee, points_img, camera->color_intrinsic_matrix(),
                   camera->color_distortion_coeffs(), rvec, tvec,
                   cv::SOLVEPNP_SQPNP);

      // Convert position.
      const Eigen::Translation3d p_world_to_camera(tvec[0], tvec[1], tvec[2]);

      // Convert angle.
      const Eigen::Vector3d aa_vec(rvec[0], rvec[1], rvec[2]);
      const double angle = aa_vec.norm();
      const Eigen::Vector3d axis =
          angle != 0 ? (aa_vec / angle).eval() : Eigen::Vector3d::Identity();
      const Eigen::AngleAxisd aa_world_to_camera(angle, axis);

      // Invert transformation.
      const Eigen::Isometry3d T_world_to_camera =
          p_world_to_camera * aa_world_to_camera;
      const Eigen::Isometry3d T_camera_to_world = T_world_to_camera.inverse();

      // Extract position and quaternion.
      const Eigen::Vector3d pos = T_camera_to_world.translation();
      const Eigen::Quaterniond quat(T_camera_to_world.linear());

      // Send camera pose to Redis.
      redis.set(KEY_POS, pos);
      redis.set(KEY_ORI, quat);
      redis.commit();

      std::cout << "Done." << std::endl
                << std::endl
                << "ee positions:" << std::endl;
      for (const cv::Point3d& point : points_ee) {
        std::cout << point << std::endl;
      }
      std::cout << std::endl << "image points:" << std::endl;
      for (const cv::Point2d& point : points_img) {
        std::cout << point << std::endl;
      }
      std::cout << std::endl
                << ctrl_utils::bold << "rvec" << ctrl_utils::normal << ": "
                << rvec << std::endl
                << ctrl_utils::bold << "tvec" << ctrl_utils::normal << ": "
                << tvec << std::endl
                << std::endl
                << "Sending to Redis:" << std::endl
                << ctrl_utils::bold << KEY_POS << ctrl_utils::normal << ": "
                << pos << std::endl
                << ctrl_utils::bold << KEY_ORI << ctrl_utils::normal << ": "
                << quat.coeffs() << std::endl;
    }
  });

  // Create image window.
  cv::namedWindow(camera->name());
  cv::setMouseCallback(
      camera->name(),
      [](int event, int x, int y, int flags, void* userdata) {
        // Trigger on left button up.
        if (event != cv::EVENT_LBUTTONUP) return;

        CameraRobotState& state =
            *reinterpret_cast<CameraRobotState*>(userdata);

        // Send click pose.
        state.ownership.acquire();
        if (!g_runloop) return;

        state.points_ee.push_back(state.point_ee);
        state.points_img.emplace_back(x, y);
        state.data_ready.release();
      },
      &state);

  // Receive frames at 30 fps.
  ctrl_utils::Timer timer(30);
  while (g_runloop) {
    timer.Sleep();

    // Request end-effector position from Redis.
    std::future<Eigen::Vector3d> fut_pos_ee =
        redis.get<Eigen::Vector3d>(args->key_pos);
    redis.commit();

    // Get color and depth images.
    cv::Mat img_color = camera->color_image();
    cv::Mat img_depth = camera->depth_image();

    // Register depth image to color.
    if (kinect2 != nullptr) {
      kinect2->RegisterDepthToColor(img_depth, img_depth_registered);
    } else {
      throw std::runtime_error("Not implemented yet.");
    }

    // Zero out unseen depth pixels.
    for (int i = 0; i < camera->color_height(); i++) {
      for (int j = 0; j < camera->color_width(); j++) {
        if (img_depth_registered.at<float>(i, j) > 0) continue;
        img_color.at<cv::Vec3b>(i, j) = 0;
      }
    }

    // Get end-effector position.
    const Eigen::Vector3d pos_ee = fut_pos_ee.get();
    const cv::Point3d point_ee(pos_ee(0), pos_ee(1), pos_ee(2));

    // If calibration isn't ready, try next cycle.
    if (!state.ownership.try_acquire()) continue;

    // Update state.
    state.point_ee = point_ee;
    std::swap(img_color, state.img_color);
    std::swap(img_depth_registered, state.img_depth);

    // Show image.
    cv::imshow(camera->name(), state.img_color);
    cv::imshow(camera->name() + "depth", state.img_depth);

    // Allow click to process with updated image..
    state.ownership.release();

    if (cv::waitKey(1) == 'q') {
      g_runloop = false;
    }
  }

  // Shut down.
  std::cout << std::endl << "Stopping calibration thread... " << std::flush;
  state.ownership.release();
  state.data_ready.release();
  if (thread_calibrate.joinable()) {
    thread_calibrate.join();
  }
  std::cout << "Done." << std::endl;

  std::cout << "Shutting down camera..." << std::endl;

  return 0;
}
