/**
 * main.cc
 *
 * Copyright 2019. All Rights Reserved.
 *
 * Created: July 29, 2019
 * Authors: Toki Migimatsu
 */

// std
#include <csignal>    // std::signal, std::sig_atomic_t
#include <exception>  // std::invalid_argument
#include <iostream>   // std::cout

// external
#include <ctrl_utils/argparse.h>
#include <ctrl_utils/eigen_string.h>
#include <ctrl_utils/opencv.h>
#include <ctrl_utils/redis_client.h>
#include <ctrl_utils/thread_pool.h>
#include <ctrl_utils/timer.h>
#include <redis_gl/redis_gl.h>
#include <redis_rgbd/kinect2.h>

#include <Eigen/Eigen>

namespace {

volatile std::sig_atomic_t g_runloop = true;
void Stop(int signal) { g_runloop = false; }

enum class DataType { ColorImage, DepthImage };

struct Args : ctrl_utils::Args {
  explicit Args(ctrl_utils::Args&& args)
      : ctrl_utils::Args(std::move(args)), description_(CreateDescription()) {
    key_prefix = "rgbd::" + camera_name + "::";
  }

  virtual std::string_view description() const override { return description_; }

  std::string camera =
      Arg<std::string>("camera", "One of {kinect, kinect2, realsense}.");

  std::string serial =
      Kwarg<std::string>("serial", "", "Camera serial number.");

  std::string redis_host =
      Kwarg<std::string>("h,redis-host", "127.0.0.1", "Redis hostname.");

  int redis_port = Kwarg<int>("p,redis-port", 6379, "Redis port.");

  std::string redis_pass =
      Kwarg<std::string>("a,redis-pass", "", "Redis password.");

  std::string camera_name = Kwarg<std::string>(
      "camera-name", "camera_0", "Camera name for the Redis key.");

  int fps = Kwarg<int>("fps", 30,
                       "Streaming fps (0 = realtime, limited by the camera).");

  bool color = Flag("color", true, "Stream color images.");

  bool depth = Flag("depth", true, "Stream depth images.");

  int res_color = Kwarg<int>("res-color", 1080, "Color image resolution.");

  bool register_depth =
      Flag("register-depth", false,
           "Streams the depth image registered to color. The depth image will "
           "be the same size as the color image.");

  bool filter_depth =
      Flag("filter-depth", false, "Filter the depth image using a box filter.");

  bool show_image = Flag("display", true, "Show image display.");

  bool use_redis_thread =
      Flag("redis-thread", false, "Run Redis in a separate thread.");

  bool verbose = Flag("verbose", false, "Print streaming frame rate.");

  std::string key_prefix;

 private:
  static std::string CreateDescription() {
    std::stringstream ss;
    ss << "Stream color and depth images to Redis." << std::endl
       << std::endl
       << "Used Redis keys:" << std::endl
       << "\t" << ctrl_utils::bold << "rgbd::<camera-name>::color"
       << ctrl_utils::normal << ": OpenCV color image." << std::endl
       << "\t" << ctrl_utils::bold << "rgbd::<camera-name>::depth"
       << ctrl_utils::normal << ": OpenCV depth image." << std::endl
       << "\t" << ctrl_utils::bold << "rgbd::<camera-name>::color::intrinsic"
       << ctrl_utils::normal << ": Color intrinsic matrix." << std::endl
       << "\t" << ctrl_utils::bold << "rgbd::<camera-name>::depth::intrinsic"
       << ctrl_utils::normal << ": Depth intrinsic matrix.";
    return ss.str();
  }

  std::string description_;
};

/**
 * Registers the camera in redis-gl.
 */
void RegisterRedisGl(const std::optional<Args>& args,
                     ctrl_utils::RedisClient& redis) {
  const redis_gl::simulator::ModelKeys model_keys("rgbd");
  redis_gl::simulator::CameraModel camera_model;
  camera_model.name = args->camera_name;
  camera_model.key_pos = args->key_prefix + "pos";
  camera_model.key_ori = args->key_prefix + "ori";
  camera_model.key_intrinsic = args->key_prefix + "color::intrinsic";
  camera_model.key_depth_image = args->key_prefix + "depth";
  camera_model.key_color_image = args->key_prefix + "color";

  redis_gl::simulator::RegisterModelKeys(redis, model_keys);
  redis_gl::simulator::RegisterCamera(redis, model_keys, camera_model);
  redis.commit();
}

/**
 * Streams camera data using realtime callback functions.
 */
void StreamRealtime(const std::optional<Args>& args,
                    std::unique_ptr<redis_rgbd::Camera>&& camera,
                    ctrl_utils::RedisClient& redis) {
  const std::string key_color = args->key_prefix + "color";
  const std::string key_depth = args->key_prefix + "depth";

  // Set up callbacks.
  camera->SetColorCallback([&key_color, &redis](cv::Mat img) {
    redis.set(key_color, img);
    redis.commit();
  });
  camera->SetDepthCallback([key_depth, &redis](cv::Mat img) {
    redis.set(key_depth, img);
    redis.commit();
  });

  // Start listening.
  camera->Start(args->color, args->depth);

  // Spin loop with 100ms sleep interval until ctrl-c.
  ctrl_utils::Timer timer(100);
  while (g_runloop) {
    timer.Sleep();
  }

  std::cout << std::endl << "Shutting down camera..." << std::endl;
}

/**
 * Push items to the queue one at a time and then pop a group at a time.
 *
 * Used to process color and depth images in parallel and send them in a batch.
 */
template <typename T>
class BatchQueue : private ctrl_utils::AtomicBuffer<T> {
 protected:
  using ctrl_utils::AtomicBuffer<T>::queue_;
  using ctrl_utils::AtomicBuffer<T>::m_;
  using ctrl_utils::AtomicBuffer<T>::cv_;
  using ctrl_utils::AtomicBuffer<T>::terminate_;

 public:
  explicit BatchQueue(size_t size_batch)
      : ctrl_utils::AtomicBuffer<T>(0), size_batch_(size_batch) {
    queue_.reserve(size_batch_);
    batch_.reserve(size_batch_);
  }

  std::vector<T> Pop() {
    std::unique_lock<std::mutex> lock(m_batch_);
    cv_batch_.wait(lock, [this]() { return !batch_.empty() || terminate_; });
    if (terminate_) return {};

    return std::move(batch_);
  }

  void Push(T&& item) {
    {
      std::unique_lock<std::mutex> lock(m_);
      queue_.push_back(std::move(item));
      if (queue_.size() >= size_batch_) {
        {
          std::lock_guard<std::mutex> lock_batch(m_batch_);
          std::swap(queue_, batch_);
        }
        cv_batch_.notify_one();
        queue_.clear();
        queue_.reserve(size_batch_);
      }
    }
    cv_.notify_one();
  }

  void Terminate() {
    terminate_ = true;
    cv_.notify_all();
    cv_batch_.notify_all();
  }

 protected:
  std::vector<T> batch_;
  std::mutex m_batch_;
  std::condition_variable cv_batch_;

  size_t size_batch_;
};

/**
 * Creates the lambda function for sending color and depth images to Redis.
 */
std::function<void()> CreateSendRedisFunction(
    const std::optional<Args>& args, ctrl_utils::RedisClient& redis,
    BatchQueue<std::pair<DataType, std::string>>& redis_requests) {
  return [key_color = args->key_prefix + "color",
          key_depth = args->key_prefix + "depth", &redis, &redis_requests]() {
    std::vector<std::pair<DataType, std::string>> batch = redis_requests.Pop();
    for (std::pair<DataType, std::string>& type_val : batch) {
      switch (type_val.first) {
        case DataType::ColorImage:
          redis.set(key_color, std::move(type_val.second));
          break;
        case DataType::DepthImage:
          redis.set(key_depth, std::move(type_val.second));
          break;
      }
    }
    redis.commit();
  };
}

/**
 * Preallocates the color image according to the command line args and returns
 * the associated intrinsic matrix.
 */
std::pair<cv::Mat, Eigen::Matrix3f> PrepareColorImage(
    const std::optional<Args>& args,
    const std::unique_ptr<redis_rgbd::Camera>& camera) {
  Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> color_intrinsic(
      reinterpret_cast<float*>(camera->color_intrinsic_matrix().data));

  // Unscaled color image.
  if (args->res_color == camera->color_height()) {
    return std::make_pair(cv::Mat(), color_intrinsic);
  }

  // Scaled color image.
  const int rows = args->res_color;
  const double scale = static_cast<double>(rows) / camera->color_height();
  const int cols = camera->color_width() * scale + 0.5;
  return std::make_pair(cv::Mat(rows, cols, CV_32FC1), scale * color_intrinsic);
}

/**
 * Preallocates the depth image according to the command line args and returns
 * the associated intrinsic matrix.
 */
std::pair<cv::Mat, Eigen::Matrix3f> PrepareDepthImage(
    const std::optional<Args>& args,
    const std::unique_ptr<redis_rgbd::Camera>& camera) {
  // Unregistered depth image.
  if (!args->register_depth) {
    Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>
        depth_intrinsic(
            reinterpret_cast<float*>(camera->depth_intrinsic_matrix().data));
    return std::make_pair(cv::Mat(), depth_intrinsic);
  }

  Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> color_intrinsic(
      reinterpret_cast<float*>(camera->color_intrinsic_matrix().data));

  // Unscaled registered depth image.
  if (args->res_color == camera->color_height()) {
    return std::make_pair(cv::Mat(), color_intrinsic);
  }

  // Scaled registered depth image.
  const int rows = args->res_color;
  const double scale = static_cast<double>(rows) / camera->color_height();
  const int cols = camera->color_width() * scale + 0.5;
  return std::make_pair(cv::Mat(rows, cols, CV_32FC1), scale * color_intrinsic);
}

/**
 * Creates the lambda function for encoding and pushing a color image to the
 * Redis queue.
 */
std::function<void()> CreateProcessColorFunction(
    const std::optional<Args>& args,
    const std::unique_ptr<redis_rgbd::Camera>& camera, cv::Mat& img_color,
    BatchQueue<std::pair<DataType, std::string>>& redis_requests) {
  return [&args, &camera, &img_color, &redis_requests]() mutable {
    // Get color image.
    cv::Mat img_color_raw = camera->color_image();

    if (args->res_color != camera->color_height()) {
      // Resize image.
      cv::resize(img_color_raw, img_color, img_color.size(), 0, 0,
                 cv::INTER_CUBIC);
    } else {
      img_color = img_color_raw;
    }

    // Send image string.
    redis_requests.Push(
        std::make_pair(DataType::ColorImage, ctrl_utils::ToString(img_color)));
  };
}

/**
 * Creates the lambda function for encoding and pushing a depth image to the
 * Redis queue.
 */
std::function<void()> CreateProcessDepthFunction(
    const std::optional<Args>& args,
    const std::unique_ptr<redis_rgbd::Camera>& camera, cv::Mat& img_depth,
    BatchQueue<std::pair<DataType, std::string>>& redis_requests) {
  const auto* kinect2 = dynamic_cast<redis_rgbd::Kinect2*>(camera.get());
  return [&args, &camera, &img_depth, img_depth_reg = cv::Mat(),
          img_depth_blur = cv::Mat(), kinect2, &redis_requests]() mutable {
    // Get depth image.
    cv::Mat img_depth_raw = camera->depth_image();

    if (args->register_depth) {
      // Register depth image.
      kinect2->RegisterDepthToColor(img_depth_raw, img_depth_reg);

      if (args->res_color != camera->color_height()) {
        // Resize image.
        cv::resize(img_depth_reg, img_depth, img_depth.size(), 0, 0,
                   cv::INTER_NEAREST);
      } else {
        img_depth = img_depth_reg;
      }
    } else {
      img_depth = img_depth_raw;
    }

    if (args->filter_depth) {
      cv::medianBlur(img_depth, img_depth_blur, 5);
      for (int i = 0; i < img_depth_blur.rows; i++) {
        for (int j = 0; j < img_depth_blur.cols; j++) {
          float& d = img_depth_blur.at<float>(i, j);
          if (d < 400 || d > 4000) {  // Range between 40cm, 400cm
            d = 0;
          }
        }
      }
      img_depth = img_depth_blur;
    }

    // Send image string.
    redis_requests.Push(
        std::make_pair(DataType::DepthImage, ctrl_utils::ToString(img_depth)));
  };
}

/**
 * Streams images at a fixed frequency.
 */
void StreamFps(const std::optional<Args>& args,
               std::unique_ptr<redis_rgbd::Camera>&& camera,
               ctrl_utils::RedisClient& redis) {
  // Create Redis request queue.
  const size_t size_batch = static_cast<size_t>(args->color) + args->depth;
  BatchQueue<std::pair<DataType, std::string>> redis_requests(size_batch);

  // Preallocate images and publish intrinsics to Redis.
  auto [img_color, intrinsic_color] = PrepareColorImage(args, camera);
  auto [img_depth, intrinsic_depth] = PrepareDepthImage(args, camera);
  redis.set(args->key_prefix + "color::intrinsic", intrinsic_color);
  redis.set(args->key_prefix + "depth::intrinsic", intrinsic_depth);
  redis.commit();
  cv::Mat img_depth_display;

  // Create image processing functions.
  std::function<void()> ProcessColor =
      CreateProcessColorFunction(args, camera, img_color, redis_requests);
  std::function<void()> ProcessDepth =
      CreateProcessDepthFunction(args, camera, img_depth, redis_requests);

  // Create thread pool.
  ctrl_utils::ThreadPool<void> thread_pool(2);

  // Create Redis send function.
  const std::function<void()> SendRedis =
      CreateSendRedisFunction(args, redis, redis_requests);
  std::thread redis_thread;
  if (args->use_redis_thread) {
    // Start Redis thread if specified.
    redis_thread = std::thread([&SendRedis]() {
      while (g_runloop) SendRedis();
    });
  }

  // Start listening.
  camera->Start(args->color, args->depth);

  // Send frames at given fps.
  ctrl_utils::Timer timer(args->fps);
  while (g_runloop) {
    timer.Sleep();

    // Request frames.
    std::future<void> fut_color, fut_depth;
    if (args->color) {
      fut_color = thread_pool.Submit(ProcessColor);
    }
    if (args->depth) {
      fut_depth = thread_pool.Submit(ProcessDepth);
    }

    // Wait for results.
    if (args->color) {
      fut_color.wait();
    }
    if (args->depth) {
      fut_depth.wait();
    }

    // Send frames.
    if (!args->use_redis_thread) {
      SendRedis();
    }

    if (args->verbose) {
      std::cout << timer.num_iters() << ": " << timer.time_elapsed() << "s "
                << timer.average_freq() << "Hz" << std::endl;
    }

    if (args->show_image) {
      if (args->color) {
        cv::imshow("Color", img_color);
      }
      if (args->depth) {
        cv::normalize(img_depth, img_depth_display, 1.0, 0.0, cv::NORM_INF);
        cv::imshow("Depth", img_depth_display);
      }
      cv::waitKey(1);
    }
  }

  // Terminate Redis requests.
  redis_requests.Terminate();
  if (redis_thread.joinable()) {
    redis_thread.join();
  }

  std::cout << std::endl << "Shutting down camera..." << std::endl;
}

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

  std::unique_ptr<redis_rgbd::Camera> camera;
  if (args->camera == "kinect2") {
    camera = std::make_unique<redis_rgbd::Kinect2>(args->verbose);
  } else {
    std::cerr << args->camera << " is not supported." << std::endl;
    return 1;
  }

  const bool is_connected = camera->Connect(args->serial);
  if (!is_connected) return 1;
  std::cout << "Done." << std::endl;

  // Connect to Redis.
  std::cout << "Connecting to Redis server at " << args->redis_host << ":"
            << args->redis_port << "... " << std::flush;

  ctrl_utils::RedisClient redis;
  redis.connect(args->redis_host, args->redis_port, args->redis_pass);
  std::cout << "Done." << std::endl;

  // Register camera in redis-gl.
  RegisterRedisGl(args, redis);

  // Start streaming.
  std::cout << "Streaming..." << std::endl;
  if (args->fps == 0) {
    StreamRealtime(args, std::move(camera), redis);
  } else {
    StreamFps(args, std::move(camera), redis);
  }

  return 0;
}
