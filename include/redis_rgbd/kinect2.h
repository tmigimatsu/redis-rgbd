/**
 * kinect2.h
 *
 * Copyright 2021. All Rights Reserved.
 *
 * Created: May 20, 2020
 * Authors: Toki Migimatsu
 */

#ifndef REDIS_RGBD_STREAM_KINECT2_H_
#define REDIS_RGBD_STREAM_KINECT2_H_

#include <memory>  // std::unique_ptr

#include "camera.h"

namespace libfreenect2 {

class Freenect2;
class Freenect2Device;

}  // namespace libfreenect2

namespace redis_rgbd {

class Kinect2 : public Camera {
 public:
  static constexpr size_t kColorWidth = 1920;
  static constexpr size_t kColorHeight = 1080;

  static constexpr size_t kDepthWidth = 512;
  static constexpr size_t kDepthHeight = 424;

  Kinect2();

  virtual ~Kinect2();

  /**
   * Connects to the Kinect usb device.
   *
   * @param serial Optional serial address.
   * @returns True if the device was successfully connected, false otherwise.
   */
  virtual bool Connect(const std::string& serial = "") override;

  /**
   * Starts streaming image frames.
   *
   * @param color Whether to stream color frames.
   * @param depth Whether to stream depth frames.
   */
  virtual void Start(bool color, bool depth) override;

  /**
   * Stops streaming image frames.
   */
  virtual void Stop() override;

  /**
   * Assigns a callback function to be called when a color image is received.
   */
  virtual void SetColorCallback(
      std::function<void(cv::Mat)>&& callback) override;

  /**
   * Assigns a callback function to be called when a depth image is received.
   */
  virtual void SetDepthCallback(
      std::function<void(cv::Mat)>&& callback) override;

  /**
   * Returns a uint8 BGR image.
   */
  virtual cv::Mat color_image() const override;

  /**
   * Returns a float32 depth image.
   */
  virtual cv::Mat depth_image() const override;

  virtual size_t color_width() const override { return kColorWidth; }
  virtual size_t color_height() const override { return kColorHeight; }
  virtual size_t depth_width() const override { return kDepthWidth; }
  virtual size_t depth_height() const override { return kDepthHeight; }

 private:
  class Kinect2Impl;
  std::unique_ptr<Kinect2Impl> impl_;
};

// protected : class FrameListener : public libfreenect2::FrameListener {
//  public:
//   virtual bool onNewFrame(libfreenect2::Frame::Type type,
//                           libfreenect2::Frame* frame) override;

//   std::function<void(const uint32_t*)> rgb_callback_;
//   std::function<void(const float*)> depth_callback_;

//   std::unique_ptr<libfreenect2::Frame> frame_color_;
//   std::unique_ptr<libfreenect2::Frame> frame_depth_undistorted_;
//   std::unique_ptr<libfreenect2::Frame> frame_color_registered_;
//   std::unique_ptr<libfreenect2::Frame> frame_depth_big_;

//   std::unique_ptr<libfreenect2::Registration> registration_;
// };

// void RgbCallback(const uint32_t* buffer);

// void DepthCallback(const float* buffer);

// libfreenect2::Freenect2 freenect_;
// libfreenect2::Freenect2Device* dev_;
// FrameListener listener_;

// std::mutex mtx_rgb_;
// std::vector<uint8_t> buffer_rgb_ = std::vector<uint8_t>(3 * kSizeRgb);
// std::unique_ptr<std::promise<cv::Mat>> promise_rgb_;

// std::mutex mtx_depth_;
// std::vector<float> buffer_depth_ = std::vector<float>(kSizeDepth);
// std::vector<float> buffer_depth_big_ = std::vector<float>(kSizeRgb);
// std::unique_ptr<std::promise<cv::Mat>> promise_depth_;

// std::pair<cv::Mat, cv::Mat> distortion_maps_;

// bool rgb_;
// bool depth_;

// ctrl_utils::RedisClient redis_;
// };

}  // namespace redis_rgbd

#endif  // REDIS_RGBD_STREAM_CAMERA_H_
