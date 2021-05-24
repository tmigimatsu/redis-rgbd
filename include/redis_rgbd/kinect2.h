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

  // From Kinect2 device.
  static constexpr std::array<float, 9> kColorIntrinsicMatrix = {
      1081.3720703125,  // [[ fx ,
      0,                //    0  ,
      959.5,            //    cx ],
      0,                //  [ 0  ,
      1081.3720703125,  //    fy ,
      539.5,            //    cy ],
      0,                //  [ 0 ,
      0,                //    0 ,
      1,                //    1 ]]
  };

  // From calibration on 09/04/19.
  // static constexpr std::array<float, 9> kColorInstrinsicMatrix = {
  //     1.04557812e+03,
  //     0.,
  //     9.58597924e+02,
  //     0.,
  //     1.04691518e+03,
  //     5.40586651e+02,
  //     0.,
  //     0.,
  //     1.,
  // };

  // From calibration on 09/04/19.
  static constexpr std::array<float, 5> kColorDistortionCoeffs = {
      0.03519172,  // [ k1 ,
      0.00169421,  //   k2 ,
      0.00131543,  //   p1 ,
      -0.0004966,  //   p2 ,
      -0.06287118  //   k3 ]
  };

  // From Kinect2 device.
  static constexpr std::array<float, 9> kDepthIntrinsicMatrix = {
      364.97171020507812,  // [[ fx ,
      0,                   //    0  ,
      259.31719970703125,  //    cx ],
      0,                   //  [ 0  ,
      364.97171020507812,  //    fy ,
      204.24639892578125,  //    cy ],
      0,                   //  [ 0 ,
      0,                   //    0 ,
      1,                   //    1 ]]
  };

  // From Kinect2 device.
  static constexpr std::array<float, 5> kDepthDistortionCoeffs = {
      0.091797769069671631,  // [ k1 ,
      -0.2636851966381073,   //   k2 ,
      0,                     //   p1 ,
      0,                     //   p2 ,
      0.089108601212501526   //   k3 ]
  };

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
   * Gets a color frame as a uint8 BGR image.
   */
  virtual cv::Mat color_image() const override;

  /**
   * Gets a depth frame as a float32 depth image.
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
   * Color intrinsic matrix coefficients [fx, 0, cx, 0, fy, cy, 0, 0, 1].
   */
  virtual const std::array<float, 9>& color_intrinsic_matrix() const override {
    return kColorIntrinsicMatrix;
  };

  /**
   * Color distortion coefficients [k1, k2, p1, p2, k3].
   */
  virtual const std::array<float, 5>& color_distortion_coeffs() const override {
    return kColorDistortionCoeffs;
  };

  /**
   * Width of the depth image.
   */
  virtual size_t depth_width() const override { return kDepthWidth; }

  /**
   * Height of the depth image.
   */
  virtual size_t depth_height() const override { return kDepthHeight; }

  /**
   * Depth intrinsic matrix coefficients [fx, 0, cx, 0, fy, cy, 0, 0, 1].
   */
  virtual const std::array<float, 9>& depth_intrinsic_matrix() const override {
    return kDepthIntrinsicMatrix;
  };

  /**
   * Depth distortion coefficients [k1, k2, p1, p2, k3].
   */
  virtual const std::array<float, 5>& depth_distortion_coeffs() const override {
    return kDepthDistortionCoeffs;
  };

  /**
   * Undistort the depth image and register the color image with it.
   *
   * @param color 1920 x 1080 uint8 BGR image.
   * @param depth 512 x 424 float32 depth image.
   * @param color_out 512 x 424 uint8 registered BGR image (if not null).
   * @param depth_out 512 x 424 float32 undistorted depth image (if not null).
   */
  static void RegisterColorDepth(const cv::Mat& color, const cv::Mat& depth,
                                 cv::Mat* color_out = nullptr,
                                 cv::Mat* depth_out = nullptr);

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
