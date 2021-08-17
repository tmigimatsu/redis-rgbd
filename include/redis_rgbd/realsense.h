/**
 * realsense.h
 *
 * Copyright 2019. All Rights Reserved.
 *
 * Created: July 29, 2019
 * Authors: Toki Migimatsu
 */

#ifndef REDIS_RGBD_REALSENSE_H_
#define REDIS_RGBD_REALSENSE_H_

#include <atomic>  // std::atomic_bool
#include <future>  // std::future
#include <memory>  // std::unique_ptr
#include <mutex>   // std::mutex
#include <thread>  // std::thread
#include <vector>  // std::vector

#include "camera.h"

namespace redis_rgbd {

class RealSense : public Camera {
 public:
  static const std::string kName;  // realsense

  static constexpr size_t kColorHeight = 480;
  static constexpr size_t kColorWidth = 640;
  static constexpr size_t kColorChannel = CV_8UC3;

  static constexpr size_t kDepthHeight = 480;
  static constexpr size_t kDepthWidth = 640;
  static constexpr size_t kDepthChannel = CV_32FC1;

  RealSense();

  virtual ~RealSense();

  /**
   * Camera name.
   */
  virtual const std::string& name() const override { return kName; }

  /**
   * Connects to the RealSense usb device.
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
      std::function<void(cv::Mat)>&& callback) override {}

  /**
   * Assigns a callback function to be called when a depth image is received.
   */
  virtual void SetDepthCallback(
      std::function<void(cv::Mat)>&& callback) override {}

  /**
   * Gets a color frame as a uint8 BGR image. This image can be modified.
   */
  virtual cv::Mat color_image() const override;

  /**
   * Gets a depth frame as a float32 depth image. This image can be modified.
   */
  virtual cv::Mat depth_image() const override;

  /**
   * Width of the color image.
   */
  virtual size_t color_width() const override { return kColorWidth; }

  /**
   * Height of the color image.
   */
  virtual size_t color_height() const override { return kColorHeight; }

  /**
   * Channel type of the color image.
   */
  virtual int color_channel() const override { return kColorChannel; }

  /**
   * Color intrinsic matrix coefficients [fx, 0, cx; 0, fy, cy; 0, 0, 1].
   */
  virtual const cv::Mat& color_intrinsic_matrix() const override;

  /**
   * Color distortion coefficients [k1, k2, p1, p2, k3].
   */
  virtual const std::array<float, 5>& color_distortion_coeffs() const override;

  /**
   * Width of the depth image.
   */
  virtual size_t depth_width() const override { return kDepthWidth; }

  /**
   * Height of the depth image.
   */
  virtual size_t depth_height() const override { return kDepthHeight; }

  /**
   * Channel type of the depth image.
   */
  virtual int depth_channel() const override { return kDepthChannel; }

  /**
   * Depth intrinsic matrix coefficients [fx, 0, cx; 0, fy, cy; 0, 0, 1].
   */
  virtual const cv::Mat& depth_intrinsic_matrix() const override;

  /**
   * Depth distortion coefficients [k1, k2, p1, p2, k3].
   */
  virtual const std::array<float, 5>& depth_distortion_coeffs() const override;

 protected:
  class RealSenseImpl;
  std::unique_ptr<RealSenseImpl> impl_;
};

}  // namespace redis_rgbd

#endif  // REDIS_RGBD_REALSENSE_H_
