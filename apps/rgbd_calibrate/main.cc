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
};

struct CameraRobotState {
  cv::Mat img_color;
  cv::Mat img_depth;
  cv::Point3d pos_ee;
  cv::Point2d point_img;
  std::binary_semaphore ownership = std::binary_semaphore(false);
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
    camera = std::make_shared<redis_rgbd::Kinect2>();
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
  state.img_depth = cv::Mat(camera->color_width(), camera->color_height(),
                            camera->depth_channel());
  cv::Mat img_bgr_flipped = cv::Mat(
      camera->color_width(), camera->color_height(), camera->color_channel());

  std::thread thread_calibrate([camera, &state]() {
    std::vector<cv::Point3d> points_ee;
    std::vector<cv::Point2d> points_img;

    while (g_runloop) {
      // Get data when available.
      state.data_ready.acquire();
      points_ee.push_back(state.pos_ee);
      points_img.push_back(state.point_img);
      state.ownership.release();

      if (points_ee.size() < 3) continue;

      // Optimize.
      cv::Mat rvec, tvec;
      // TODO: Fix intrinsic matrix.
      cv::solvePnP(points_ee, points_img, camera->color_intrinsic_matrix(),
                   camera->color_distortion_coeffs(), rvec, tvec);

      std::cout << "ee positions:" << std::endl;
      for (const cv::Point3d& point : points_ee) {
      std::cout << point << std::endl;
      }
      std::cout << "image points:" << std::endl;
      for (const cv::Point2d& point : points_img) {
      std::cout << point << std::endl;
      }
      std::cout << "rvec: " << rvec << std::endl;
      std::cout << "tvec: " << tvec << std::endl;
    }
  });
  std::vector<cv::Point2d> image_points;

  // Create image window.
  cv::namedWindow(camera->name());
  cv::setMouseCallback(
      camera->name(),
      [](int event, int x, int y, int flags, void* userdata) {
        if (event != cv::EVENT_LBUTTONUP) return;

        CameraRobotState& state =
            *reinterpret_cast<CameraRobotState*>(userdata);
        state.ownership.acquire();
        state.point_img.x = x;
        state.point_img.y = y;
        state.data_ready.release();
      },
      &state);

  // Receive frames at 30 fps.
  ctrl_utils::Timer timer(30);
  while (g_runloop) {
    timer.Sleep();

    if (!state.ownership.try_acquire()) continue;

    // Get end-effector position from Redis.
    std::future<Eigen::Vector3d> fut_pos_ee =
        redis.get<Eigen::Vector3d>(args->key_pos);
    redis.commit();

    // Get color and depth images.
    state.img_color = camera->color_image();
    cv::Mat img_depth = camera->depth_image();

    // Register depth image to color.
    if (kinect2 != nullptr) {
      kinect2->RegisterDepthToColor(img_depth, state.img_color,
                                    state.img_depth);
    } else {
      throw std::runtime_error("Not implemented yet.");
    }

    // Zero out unseen depth pixels.
    for (int i = 0; i < camera->color_height(); i++) {
      for (int j = 0; j < camera->color_width(); j++) {
        if (state.img_depth.at<float>(i, j) <= 0) {
          cv::Vec3b pixel = state.img_color.at<cv::Vec3b>(i, j);
          pixel = 0;
        }
      }
    }

    // Get end-effector position.
    const Eigen::Vector3d pos_ee = fut_pos_ee.get();
    state.pos_ee.x = pos_ee(0);
    state.pos_ee.y = pos_ee(1);
    state.pos_ee.z = pos_ee(2);

    state.ownership.release();

    // Flip image about Y-axis before showing.
    cv::flip(state.img_color, img_bgr_flipped, 1);
    cv::imshow(camera->name(), state.img_color);
  }

  return 0;
}
