/**
 * camera.h
 *
 * Copyright 2021. All Rights Reserved.
 *
 * Created: May 20, 2021
 * Authors: Toki Migimatsu
 */

#ifndef REDIS_RGBD_STREAM_CAMERA_H_
#define REDIS_RGBD_STREAM_CAMERA_H_

#include <functional>  // std::function
#include <opencv2/opencv.hpp>
#include <string>  // std::string

namespace redis_rgbd {

class Camera {
 public:
  virtual ~Camera() = default;

  /**
   * Camera name.
   */
  virtual const std::string& name() const = 0;

  /**
   * Connects to the Kinect usb device.
   *
   * @param serial Optional serial address.
   * @returns True if the device was successfully connected, false otherwise.
   */
  virtual bool Connect(const std::string& serial = "") = 0;

  /**
   * Starts streaming image frames.
   *
   * @param color Whether to stream color frames.
   * @param depth Whether to stream depth frames.
   */
  virtual void Start(bool rgb, bool depth) = 0;

  /**
   * Stops streaming image frames.
   */
  virtual void Stop() = 0;

  /**
   * Assigns a callback function to be called when a color image is received.
   */
  virtual void SetColorCallback(std::function<void(cv::Mat)>&& callback) = 0;

  /**
   * Assigns a callback function to be called when a depth image is received.
   */
  virtual void SetDepthCallback(std::function<void(cv::Mat)>&& callback) = 0;

  /**
   * Gets a color image.
   */
  virtual cv::Mat color_image() const = 0;

  /**
   * Gets a depth image.
   */
  virtual cv::Mat depth_image() const = 0;

  /**
   * Width of the color image.
   */
  virtual size_t color_width() const = 0;

  /**
   * Height of the color image.
   */
  virtual size_t color_height() const = 0;

  /**
   * Channel type of the color image.
   */
  virtual int color_channel() const = 0;

  /**
   * Color intrinsic matrix coefficients [fx, 0, cx; 0, fy, cy; 0, 0, 1].
   */
  virtual const cv::Mat& color_intrinsic_matrix() const = 0;

  /**
   * Color distortion coefficients [k1, k2, p1, p2, k3].
   */
  virtual const std::array<float, 5>& color_distortion_coeffs() const = 0;

  /**
   * Width of the depth image.
   */
  virtual size_t depth_width() const = 0;

  /**
   * Height of the depth image.
   */
  virtual size_t depth_height() const = 0;

  /**
   * Channel type of the depth image.
   */
  virtual int depth_channel() const = 0;

  /**
   * Depth intrinsic matrix coefficients [fx, 0, cx; 0, fy, cy; 0, 0, 1].
   */
  virtual const cv::Mat& depth_intrinsic_matrix() const = 0;

  /**
   * Depth distortion coefficients [k1, k2, p1, p2, k3].
   */
  virtual const std::array<float, 5>& depth_distortion_coeffs() const = 0;
};

}  // namespace redis_rgbd

#endif  // REDIS_RGBD_STREAM_CAMERA_H_
