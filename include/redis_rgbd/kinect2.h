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
  static const std::string kName;  // kinect2

  static constexpr size_t kColorWidth = 1920;
  static constexpr size_t kColorHeight = 1080;
  static constexpr size_t kColorChannel = CV_8UC3;

  static constexpr size_t kDepthWidth = 512;
  static constexpr size_t kDepthHeight = 424;
  static constexpr size_t kDepthChannel = CV_32FC1;

  // From Kinect2 device.
  static constexpr std::array<float, 9> kColorIntrinsicMatrix = {
      1081.3720703125,      // [[ fx ,
      0,                    //    0  ,
      kColorWidth - 959.5,  //    cx ],
      0,                    //  [ 0  ,
      1081.3720703125,      //    fy ,
      539.5,                //    cy ],
      0,                    //  [ 0 ,
      0,                    //    0 ,
      1,                    //    1 ]]
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
      364.97171020507812,                // [[ fx ,
      0,                                 //    0  ,
      kDepthWidth - 259.31719970703125,  //    cx ],
      0,                                 //  [ 0  ,
      364.97171020507812,                //    fy ,
      204.24639892578125,                //    cy ],
      0,                                 //  [ 0 ,
      0,                                 //    0 ,
      1,                                 //    1 ]]
  };

  // From Kinect2 device.
  static constexpr std::array<float, 5> kDepthDistortionCoeffs = {
      0.091797769069671631,  // [ k1 ,
      -0.2636851966381073,   //   k2 ,
      0,                     //   p1 ,
      0,                     //   p2 ,
      0.089108601212501526   //   k3 ]
  };

  static constexpr std::array<float, 9> kDepthToColorIntrinsicMatrix = {
      2.93,   // [[ fx ,
      0,      //    0  ,
      150.5,  //    cx ],
      0,      //  [ 0  ,
      2.93,   //    fy ,
      -63.5,  //    cy ],
      0,      //  [ 0 ,
      0,      //    0 ,
      1,      //    1 ]]
  };

  explicit Kinect2(bool verbose = false);

  virtual ~Kinect2();

  /**
   * Camera name.
   */
  virtual const std::string& name() const override { return kName; }

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
  virtual const cv::Mat& color_intrinsic_matrix() const override {
    static const cv::Mat matrix(
        3, 3, CV_32FC1, const_cast<float*>(kColorIntrinsicMatrix.data()));
    return matrix;
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
   * Channel type of the depth image.
   */
  virtual int depth_channel() const override { return kDepthChannel; }

  /**
   * Depth intrinsic matrix coefficients [fx, 0, cx; 0, fy, cy; 0, 0, 1].
   */
  virtual const cv::Mat& depth_intrinsic_matrix() const override {
    static const cv::Mat matrix(
        3, 3, CV_32FC1, const_cast<float*>(kDepthIntrinsicMatrix.data()));
    return matrix;
  };

  /**
   * Depth distortion coefficients [k1, k2, p1, p2, k3].
   */
  virtual const std::array<float, 5>& depth_distortion_coeffs() const override {
    return kDepthDistortionCoeffs;
  };

  /**
   * Registers the color image to the depth image.
   *
   * Assumes the color and depth images are given by `color_image()`, and
   * `depth_image()`, which are flipped horizontally (as opposed
   * to the default selfie camera view given by the Kinect2 drivers).
   *
   * @param color 1920 x 1080 uint8 BGR image.
   * @param depth 512 x 424 float32 depth image.
   * @param color_out 512 x 424 uint8 registered BGR image. Will allocate a new
   *                  cv::Mat if it is not the correct shape.
   */
  static void RegisterColorToDepth(const cv::Mat& color, const cv::Mat& depth,
                                   cv::Mat& color_out);

  /**
   * Registers the depth image to the color image.
   *
   * Assumes the color and depth images are given by `color_image()`, and
   * `depth_image()`, which are flipped horizontally (as opposed
   * to the default selfie camera view given by the Kinect2 drivers).
   *
   * @param depth 512 x 424 float32 undistorted depth image.
   * @param depth_out 1920 x 1080 float32 registered depth image. Will allocate
   *                  a new cv::Mat if it is not the correct shape.
   */
  static void RegisterDepthToColor(const cv::Mat& depth, cv::Mat& depth_out);

  /**
   * Filters out noise in the depth image using a box filter.
   *
   * @param depth_reg 1920 x 1080 float32 registered depth image computed with
   *                  `Kinect2::RegisterDepthToColor()`.
   * @param depth 512 x 424 float32 depth image to be filtered in place.
   */
  static void FilterDepth(const cv::Mat& depth_reg, cv::Mat& depth);

  void SetAutoExposure(float exposure_compensation);
  void SetExposure(float integration_time_ms, float analog_gain);

 private:
  class Kinect2Impl;
  std::unique_ptr<Kinect2Impl> impl_;
};

}  // namespace redis_rgbd

#endif  // REDIS_RGBD_STREAM_CAMERA_H_
