/**
 * main.cc
 *
 * Copyright 2019. All Rights Reserved.
 *
 * Created: May 24, 2021
 * Authors: Toki Migimatsu
 */

#include <ctrl_utils/argparse.h>
#include <ctrl_utils/opencv.h>
#include <ctrl_utils/redis_client.h>
#include <ctrl_utils/timer.h>
#include <redis_rgbd/kinect2.h>

#include <csignal>    // std::signal, std::sig_atomic_t
#include <exception>  // std::invalid_argument
#include <iostream>   // std::cout

namespace {

volatile std::sig_atomic_t g_runloop = true;
void Stop(int signal) { g_runloop = false; }

struct Args : ctrl_utils::Args {
  explicit Args(ctrl_utils::Args&& args) : ctrl_utils::Args(std::move(args)) {}

  std::string redis_host =
      Kwarg<std::string>("h,redis-host", "127.0.0.1", "Redis hostname.");
  int redis_port = Kwarg<int>("p,redis-port", 6379, "Redis port.");
  std::string redis_pass =
      Kwarg<std::string>("a,redis-pass", "", "Redis password.");
  std::string key_prefix =
      Kwarg<std::string>("prefix", "rgbd::camera_0::", "Redis key prefix.");
  int fps = Kwarg<int>("fps", 30, "Streaming fps.");
};

}  // namespace

int main(int argc, char* argv[]) {
  std::signal(SIGINT, Stop);

  std::optional<Args> args = ctrl_utils::ParseArgs<Args>(argc, argv);
  if (!args.has_value()) return 1;
  std::cout << args->help_string() << std::endl << *args << std::endl;

  // Connect to Redis.
  std::cout << "Connecting to Redis server at " << args->redis_host << ":"
            << args->redis_port << "... " << std::flush;

  ctrl_utils::RedisClient redis;
  redis.connect(args->redis_host, args->redis_port, args->redis_pass);
  std::cout << "Done." << std::endl;

  const std::string key_color = args->key_prefix + "color";
  const std::string key_depth = args->key_prefix + "depth";

  // Start streaming.
  std::cout << "Listening..." << std::endl;

  // Get scaled color image first.
  cv::Mat img_bgr_mini = redis.sync_get<cv::Mat>(key_color);

  cv::Mat img_bgr(redis_rgbd::Kinect2::kColorHeight,
                  redis_rgbd::Kinect2::kColorWidth, CV_8UC3);

  cv::Mat img_depth(redis_rgbd::Kinect2::kDepthHeight,
                    redis_rgbd::Kinect2::kDepthWidth, CV_32FC1);

  cv::Mat img_bgr_reg(redis_rgbd::Kinect2::kDepthHeight,
                      redis_rgbd::Kinect2::kDepthWidth, CV_8UC3);

  ctrl_utils::Timer timer(args->fps);
  while (g_runloop) {
    timer.Sleep();
    std::future<void> fut_img_bgr_mini = redis.get(key_color, img_bgr_mini);
    std::future<void> fut_img_depth = redis.get(key_depth, img_depth);
    redis.commit();

    fut_img_bgr_mini.wait();
    cv::resize(img_bgr_mini, img_bgr, img_bgr.size());

    fut_img_depth.wait();

    redis_rgbd::Kinect2::RegisterColorToDepth(img_bgr, img_depth, img_bgr_reg);
    cv::imshow("Color", img_bgr);
    cv::imshow("Depth", img_depth);
    cv::imshow("Registered color", img_bgr_reg);
    cv::waitKey(0);
    break;
  }

  return 0;
}
