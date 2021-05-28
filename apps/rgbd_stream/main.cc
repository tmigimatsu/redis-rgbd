/**
 * main.cc
 *
 * Copyright 2019. All Rights Reserved.
 *
 * Created: July 29, 2019
 * Authors: Toki Migimatsu
 */

#include <ctrl_utils/argparse.h>
#include <ctrl_utils/opencv.h>
#include <ctrl_utils/redis_client.h>
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

  std::string camera =
      Arg<std::string>("camera", "kinect, kinect2, or realsense");

  std::string serial = Kwarg<std::string>("serial", "", "Camera serial");

  std::string redis_host =
      Kwarg<std::string>("h,redis-host", "127.0.0.1", "Redis hostname");

  int redis_port = Kwarg<int>("p,redis-port", 6379, "Redis port");

  std::string redis_pass =
      Kwarg<std::string>("a,redis-pass", "", "Redis password");

  std::string key_prefix =
      Kwarg<std::string>("prefix", "rgbd::camera_0::", "Redis key prefix");

  int fps = Kwarg<int>("fps", 30,
                       "Streaming fps (0 = realtime, limited by the camera)");

  bool color = Flag("color", true, "Stream color images");

  bool depth = Flag("depth", true, "Stream depth images");

  int res_color = Kwarg<int>("res-color", 1080, "Color image resolution");

  bool show_image = Flag("display", true, "Show image display");

  bool use_redis_thread =
      Flag("redis-thread", false, "Run Redis in a separate thread");
};

/**
 * Push items to the queue one at a time and then pop a group at a time.
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

}  // namespace

int main(int argc, char* argv[]) {
  std::signal(SIGINT, Stop);

  std::optional<Args> maybe_args = ctrl_utils::ParseArgs<Args>(argc, argv);
  if (!maybe_args.has_value()) return 1;
  const Args& args = *maybe_args;
  std::cout << args.help_string() << std::endl;
  std::cout << args << std::endl;

  // Connect to camera.
  std::cout << "Connecting to " << args.camera;
  if (!args.serial.empty()) {
    std::cout << " with serial " << args.serial;
  }
  std::cout << "... " << std::endl;

  std::unique_ptr<redis_rgbd::Camera> camera;
  if (args.camera == "kinect2") {
    camera = std::make_unique<redis_rgbd::Kinect2>();
  } else {
    std::cerr << args.camera << " is not supported." << std::endl;
    return 1;
  }

  const bool is_connected = camera->Connect(args.serial);
  if (!is_connected) return 1;
  std::cout << "Done." << std::endl;

  // Connect to Redis.
  std::cout << "Connecting to Redis server at " << args.redis_host << ":"
            << args.redis_port << "... " << std::flush;

  ctrl_utils::RedisClient redis;
  redis.connect(args.redis_host, args.redis_port, args.redis_pass);
  std::cout << "Done." << std::endl;

  const std::string key_color = args.key_prefix + "color";
  const std::string key_depth = args.key_prefix + "depth";

  // Start streaming.
  std::cout << "Streaming..." << std::endl;
  if (args.fps == 0) {
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
    camera->Start(args.color, args.depth);

    // Spin loop with 100ms sleep interval until ctrl-c.
    ctrl_utils::Timer timer(100);
    while (g_runloop) {
      timer.Sleep();
    }
  } else {
    // Create Redis request queue.
    const size_t size_batch = static_cast<size_t>(args.color) + args.depth;
    BatchQueue<std::pair<ImageType, std::string>> redis_requests(size_batch);

    // Create Redis send function.
    std::function<void()> SendRedis = [&key_color, &key_depth, &redis,
                                       &redis_requests]() {
      std::vector<std::pair<ImageType, std::string>> batch =
          redis_requests.Pop();
      for (std::pair<ImageType, std::string>& type_val : batch) {
        switch (type_val.first) {
          case ImageType::Color:
            redis.set(key_color, std::move(type_val.second));
            break;
          case ImageType::Depth:
            redis.set(key_depth, std::move(type_val.second));
            break;
        }
      }
      redis.commit();
    };

    // Create thread pool.
    ctrl_utils::ThreadPool<void> thread_pool(2);

    // Preallocate images.
    const int rows = args.res_color;
    const double scale = static_cast<double>(rows) / camera->color_height();
    const int cols = camera->color_width() * scale + 0.5;
    cv::Mat img_color, img_depth;
    cv::Mat img_color_scaled(rows, cols, CV_8UC3);

    // Create color retrieval function.
    std::function<void()> ProcessColor = [&camera, &img_color,
                                          &img_color_scaled, &key_color,
                                          &redis_requests]() {
      // Get color image.
      img_color = camera->color_image();

      // Resize image.
      if (img_color_scaled.rows != img_color.rows) {
        cv::resize(img_color, img_color_scaled, img_color_scaled.size(), 0, 0,
                   cv::INTER_CUBIC);

        // Send scaled color image.
        redis_requests.Push(std::make_pair(
            ImageType::Color, ctrl_utils::ToString(img_color_scaled)));
      } else {
        // Send color image.
        redis_requests.Push(
            std::make_pair(ImageType::Color, ctrl_utils::ToString(img_color)));
      }
    };

    // Create depth retrieval function.
    std::function<void()> ProcessDepth = [&camera, &img_depth, &key_depth,
                                          &redis_requests]() {
      // Get depth image.
      img_depth = camera->depth_image();

      // Send image string.
      redis_requests.Push(
          std::make_pair(ImageType::Depth, ctrl_utils::ToString(img_depth)));
    };

    // Start Redis thread.
    std::thread redis_thread;
    if (args.use_redis_thread) {
      redis_thread = std::thread([&SendRedis]() {
        while (g_runloop) SendRedis();
      });
    }

    // Start listening.
    camera->Start(args.color, args.depth);

    // Send frames at given fps.
    ctrl_utils::Timer timer(args.fps);
    while (g_runloop) {
      timer.Sleep();

      // Request frames.
      std::future<void> fut_color, fut_depth;
      if (args.color) {
        fut_color = thread_pool.Submit(ProcessColor);
      }
      if (args.depth) {
        fut_depth = thread_pool.Submit(ProcessDepth);
      }

      // Wait for results.
      if (args.color) {
        fut_color.wait();
      }
      if (args.depth) {
        fut_depth.wait();
      }

      // Send frames.
      if (!args.use_redis_thread) {
        SendRedis();
      }

      std::cout << timer.num_iters() << ": " << timer.time_elapsed() << "s "
                << timer.average_freq() << "Hz" << std::endl;

      if (args.show_image) {
        if (args.color) {
          cv::imshow("Color", img_color);
        }
        if (args.depth) {
          cv::imshow("Depth", img_depth);
        }
        cv::waitKey(1);
      }
    }

    // Terminate Redis requests.
    redis_requests.Terminate();
    if (redis_thread.joinable()) {
      redis_thread.join();
    }
  }

  return 0;
}
