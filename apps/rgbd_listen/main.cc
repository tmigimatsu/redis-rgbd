/**
 * main.cc
 *
 * Copyright 2019. All Rights Reserved.
 *
 * Created: May 24, 2021
 * Authors: Toki Migimatsu
 */

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

struct Args {
  std::string redis_host = "127.0.0.1";         // -h
  int redis_port = 6379;                        // -p
  std::string redis_pass = "";                  // -a
  std::string key_prefix = "rgbd::camera_0::";  // --prefix
  int fps = 30;                                 // --fps
};

Args ParseArgs(int argc, char* argv[]) {
  std::stringstream ss;
  ss << "Usage:" << std::endl
     << "\trgbd_listen" << std::endl
     << "\t\t[-h <redis hostname> (default 127.0.0.1)]" << std::endl
     << "\t\t[-p <redis port> (default 6379)]" << std::endl
     << "\t\t[-a <redis password>]" << std::endl
     << "\t\t[--prefix <redis key prefix> (default rgbd::camera_0::)]"
     << std::endl
     << "\t\t[--fps <fps> (default 30)" << std::endl
     << std::endl
     << "Example:" << std::endl
     << "\t./rgbd_listen -p 6379" << std::endl;

  Args args;
  int idx = 1;

  while (idx < argc) {
    std::string arg(argv[idx]);

    if (idx + 1 >= argc) break;
    if (arg == "-h") {
      // Redis hostname.
      args.redis_host = argv[idx + 1];
      idx += 2;
    } else if (arg == "-p") {
      // Redis port.
      args.redis_port = std::atoi(argv[idx + 1]);
      idx += 2;
    } else if (arg == "-a") {
      // Redis password.
      args.redis_pass = argv[idx + 1];
      idx += 2;
    } else if (arg == "--prefix") {
      args.key_prefix = argv[idx + 1];
      idx += 2;
    } else if (arg == "--fps") {
      args.fps = std::atoi(argv[idx + 1]);
      idx += 2;
    } else {
      // Unrecognized argument.
      break;
    }
  }

  if (idx < argc || argc <= 1) {
    // Show usage.
    throw std::invalid_argument(ss.str());
  }

  return args;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::signal(SIGINT, Stop);

  Args args;
  try {
    args = ParseArgs(argc, argv);
  } catch (std::exception& e) {
    std::cerr << e.what();
    return 1;
  }

  // Connect to Redis.
  std::cout << "Connecting to Redis server at " << args.redis_host << ":"
            << args.redis_port << "... " << std::flush;

  ctrl_utils::RedisClient redis;
  redis.connect(args.redis_host, args.redis_port, args.redis_pass);
  std::cout << "Done." << std::endl;

  const std::string key_color = args.key_prefix + "color";
  const std::string key_depth = args.key_prefix + "depth";

  // Start streaming.
  std::cout << "Listening..." << std::endl;

  cv::Mat img_bgr(redis_rgbd::Kinect2::kColorHeight,
                  redis_rgbd::Kinect2::kColorWidth, CV_8UC3);

  cv::Mat img_depth(redis_rgbd::Kinect2::kDepthHeight,
                    redis_rgbd::Kinect2::kDepthWidth, CV_32FC1);

  cv::Mat img_bgr_reg(redis_rgbd::Kinect2::kDepthHeight,
                  redis_rgbd::Kinect2::kDepthWidth, CV_8UC3);

  ctrl_utils::Timer timer(args.fps);
  while (g_runloop) {
    timer.Sleep();
    std::future<void> fut_img_bgr = redis.get(key_color, img_bgr);
    std::future<void> fut_img_depth = redis.get(key_depth, img_depth);
    redis.commit();
    fut_img_bgr.wait();
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
