/**
 * main.cc
 *
 * Copyright 2019. All Rights Reserved.
 *
 * Created: July 29, 2019
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
  std::string camera = "";
  std::string serial = "";                      // --serial
  std::string redis_host = "127.0.0.1";         // -h
  int redis_port = 6379;                        // -p
  std::string redis_pass = "";                  // -a
  std::string key_prefix = "rgbd::camera_0::";  // --prefix
  int fps = 0;                                  // --fps
};

Args ParseArgs(int argc, char* argv[]) {
  std::stringstream ss;
  ss << "Usage:" << std::endl
     << "\trgbd_stream {kinect,kinect2,realsense}" << std::endl
     << "\t\t[--serial <camera serial>]" << std::endl
     << "\t\t[-h <redis hostname> (default 127.0.0.1)]" << std::endl
     << "\t\t[-p <redis port> (default 6379)]" << std::endl
     << "\t\t[-a <redis password>]" << std::endl
     << "\t\t[--prefix <redis key prefix> (default rgbd::camera_0::)]" << std::endl
     << "\t\t[--fps <fps (0 = realtime, limited by the camera)> (default 0)"
     << std::endl
     << std::endl
     << "Example:" << std::endl
     << "\t./rgbd_stream kinect2 --fps 30" << std::endl;

  Args args;
  int idx = 1;
  if (idx < argc) {
    // Camera model.
    args.camera = argv[idx];
    idx++;
  }

  while (idx < argc) {
    std::string arg(argv[idx]);

    if (idx + 1 >= argc) break;
    if (arg == "--serial") {
      args.serial = argv[idx + 1];
      idx += 2;
    } else if (arg == "-h") {
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
    camera->Start(true, true);

    // Spin loop with 100ms sleep interval until ctrl-c.
    ctrl_utils::Timer timer(100);
    while (g_runloop) {
      timer.Sleep();
    }
  } else {
    // Start listening.
    camera->Start(true, true);

    // Send frames at given fps.
    ctrl_utils::Timer timer(args.fps);
    while (g_runloop) {
      timer.Sleep();
      redis.set(key_color, camera->color_image());
      redis.set(key_depth, camera->depth_image());
      redis.commit();
    }
  }

  // Gracefully stop the camera.
  camera->Stop();

  return 0;
}
