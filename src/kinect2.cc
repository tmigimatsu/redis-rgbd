/**
 * kinect2.cc
 *
 * Copyright 2021. All Rights Reserved.
 *
 * Created: May 20, 2021
 * Authors: Toki Migimatsu
 */

#include "redis_rgbd/kinect2.h"

#include <ctrl_utils/semaphore.h>

#include <atomic>   // std::atomic
#include <cstring>  // std::memcpy
#include <libfreenect2/frame_listener.hpp>
#include <libfreenect2/libfreenect2.hpp>
#include <opencv2/opencv.hpp>
#include <optional>  // std::optional
#include <thread>    // std::thread

namespace {

static constexpr size_t kColorHeight = redis_rgbd::Kinect2::kColorHeight;
static constexpr size_t kColorWidth = redis_rgbd::Kinect2::kColorWidth;
static constexpr size_t kColorBytes =
    kColorHeight * kColorWidth * sizeof(uint32_t);

static constexpr size_t kDepthHeight = redis_rgbd::Kinect2::kDepthHeight;
static constexpr size_t kDepthWidth = redis_rgbd::Kinect2::kDepthWidth;
static constexpr size_t kDepthBytes =
    kDepthHeight * kDepthWidth * sizeof(float);

static constexpr std::array<float, 9> kColorIntrinsicMatrix =
    redis_rgbd::Kinect2::kColorIntrinsicMatrix;

static constexpr std::array<float, 9> kDepthIntrinsicMatrix =
    redis_rgbd::Kinect2::kDepthIntrinsicMatrix;

static constexpr std::array<float, 5> kDepthDistortionCoeffs =
    redis_rgbd::Kinect2::kDepthDistortionCoeffs;

static constexpr std::array<float, 10> kColorExtrinsicX = {
    0.000574019,   // x3y0: xxx
    -2.13737e-07,  // x0y3: yyy
    4.55302e-05,   // x2y1: xxy
    0.000468875,   // x1y2: yyx
    9.54373e-05,   // x2y0: xx
    0.000142223,   // x0y2: yy
    0.000370165,   // x1y1: xy
    0.642408,      // x1y0: x
    0.00718498,    // x0y1: y
    0.173172,      // x0y0: 1
};

static constexpr std::array<float, 10> kColorExtrinsicY = {
    -1.84482e-05,  // x3y0: xxx
    0.000785405,   // x0y3: yyy
    0.000551599,   // x2y1: xxy
    3.57664e-05,   // x1y2: yyx
    0.000114572,   // x2y0: xx
    0.000319694,   // x0y2: yy
    0.000101136,   // x1y1: xy
    -0.00717631,   // x1y0: y
    0.641981,      // x0y1: x
    -0.0114988,    // x0y0: 1
};

// From Kinect2 device.
static constexpr float kShiftM = 52;
static constexpr float kShiftD = 863;

// From libfreenect2/src/registration.cpp.
static constexpr float kDepthQ = 0.01;
static constexpr float kColorQ = 0.002199;

constexpr float ComputeW(float mx, float my, std::array<float, 10> params) {
  return (mx * mx * mx * std::get<0>(params)) +
         (my * my * my * std::get<1>(params)) +
         (mx * mx * my * std::get<2>(params)) +
         (my * my * mx * std::get<3>(params)) +
         (mx * mx * std::get<4>(params)) + (my * my * std::get<5>(params)) +
         (mx * my * std::get<6>(params)) + (mx * std::get<7>(params)) +
         (my * std::get<8>(params)) + (std::get<9>(params));
}

using Arr2f = std::array<float, 2>;
constexpr Arr2f DepthToColor(float x, float y) {
  const float fx = std::get<0>(kDepthIntrinsicMatrix);
  const float cx = std::get<2>(kDepthIntrinsicMatrix);
  const float cy = std::get<5>(kDepthIntrinsicMatrix);

  const float mx = (x - cx) * kDepthQ;
  const float my = (y - cy) * kDepthQ;

  const float wx = ComputeW(mx, my, kColorExtrinsicX);
  const float wy = ComputeW(mx, my, kColorExtrinsicY);

  const float rx = (wx / (fx * kColorQ)) - (kShiftM / kShiftD);
  const float ry = (wy / kColorQ) + cy;

  return {rx, ry};
}

using DepthMap = std::array<std::array<Arr2f, kDepthHeight>, kDepthWidth>;
constexpr DepthMap ComputeDepthColorMap() {
  DepthMap depth_color_map = {0};
  for (size_t y = 0; y < std::size(depth_color_map); y++) {
    for (size_t x = 0; x < std::size(depth_color_map[y]); x++) {
      depth_color_map[y][x] = DepthToColor(x, y);
    }
  }
  return depth_color_map;
}

static constexpr DepthMap kDepthColorMap = ComputeDepthColorMap();

static std::optional<std::array<int, 2>> DepthToColorIndices(int x_depth,
                                                             int y_depth,
                                                             float z) {
  const float map_y = kDepthColorMap[y_depth][x_depth][1];
  const int ry_int = map_y + 0.5;
  if (ry_int < 0 || ry_int >= kColorHeight) return {};

  const float fx = std::get<0>(kColorIntrinsicMatrix);
  const float cx = std::get<2>(kColorIntrinsicMatrix);
  const float map_x = kDepthColorMap[y_depth][x_depth][0];

  const float rx = (map_x + (kShiftM / z)) * fx + cx;
  const int rx_int = rx + 0.5;

  return {{rx_int, ry_int}};
}

}  // namespace

namespace redis_rgbd {

class Kinect2::Kinect2Impl {
 public:
  virtual ~Kinect2Impl();

  bool Connect(const std::string& serial);

  void Start(bool color, bool depth);

  void Stop();

  void SetColorCallback(std::function<void(cv::Mat)>&& callback);

  void SetDepthCallback(std::function<void(cv::Mat)>&& callback);

  /**
   * Returns a container around locally-stored BGR image data.
   */
  cv::Mat color_image();

  /**
   * Returns a container around locally-stored depth image data.
   */
  cv::Mat depth_image();

 private:
  class FrameListener : public libfreenect2::FrameListener {
   public:
    FrameListener() : sem_color_callback(0), sem_depth_callback(0) {}

    /**
     * Copies image data from libfreenect2 to a local buffer.
     *
     * This callback function is called by libfreenect2 when a new frame is
     * received.
     */
    virtual bool onNewFrame(libfreenect2::Frame::Type type,
                            libfreenect2::Frame* frame) override;

    /**
     * Populates (and resizes) the given image with the latest color frame.
     */
    void GetBgrxFrame(cv::Mat& img);

    /**
     * Populates (and resizes) the given image with the latest depth frame.
     */
    void GetDepthFrame(cv::Mat& img);

    /**
     * Condition variable notified when new color frame is received.
     */
    std::binary_semaphore sem_color_callback;

    /**
     * Condition variable notified when new depth frame is received.
     */
    std::binary_semaphore sem_depth_callback;

   private:
    // Performs memcpy with the given mutex locked.
    static void LockCopy(void* dest, const void* src, size_t bytes,
                         std::mutex& mtx);

    // Color frame buffer populated by Kinect.
    std::array<uint32_t, kColorWidth * kColorHeight> buffer_bgrx_;
    std::mutex mtx_bgrx_;

    // Depth frame buffer populated by Kinect.
    std::array<float, kDepthWidth * kDepthHeight> buffer_depth_;
    std::mutex mtx_depth_;
  };

  libfreenect2::Freenect2 freenect_;
  FrameListener listener_;
  libfreenect2::Freenect2Device* dev_ = nullptr;

  // Boolean flag to stop callback threads.
  std::atomic<bool> is_running_;

  std::thread thread_color_callback_;
  std::thread thread_depth_callback_;

  cv::Mat img_bgr_;
  cv::Mat img_bgrx_;
  cv::Mat img_depth_;
};

/////////////
// Kinect2 //
/////////////

Kinect2::Kinect2() {}
Kinect2::~Kinect2() {}

bool Kinect2::Connect(const std::string& serial) {
  // Set up freenect and listener on first connect.
  if (!impl_) {
    impl_ = std::make_unique<Kinect2Impl>();
  }

  return impl_->Connect(serial);
}

void Kinect2::Start(bool color, bool depth) { impl_->Start(color, depth); }

void Kinect2::Stop() { impl_->Stop(); }

void Kinect2::SetColorCallback(std::function<void(cv::Mat)>&& callback) {
  impl_->SetColorCallback(std::move(callback));
}

void Kinect2::SetDepthCallback(std::function<void(cv::Mat)>&& callback) {
  impl_->SetDepthCallback(std::move(callback));
}

cv::Mat Kinect2::color_image() const { return impl_->color_image(); }

cv::Mat Kinect2::depth_image() const { return impl_->depth_image(); }

void Kinect2::RegisterColorDepth(const cv::Mat& color, const cv::Mat& depth,
                                 cv::Mat* color_out, cv::Mat* depth_out) {
  if (depth_out != nullptr) {
    cv::undistort(depth, *depth_out, kDepthIntrinsicMatrix,
                  kDepthDistortionCoeffs);
  }

  if (color_out != nullptr) {
    for (int y = 0; y < kDepthHeight; y++) {
      for (int x = 0; x < kDepthWidth; x++) {
        const std::optional<std::array<int, 2>> xy_color =
            DepthToColorIndices(x, y, depth.at<float>(y, x));
        cv::Vec3b& bgr = color_out->at<cv::Vec3b>(y, x);
        if (!xy_color.has_value()) {
          bgr[0] = 0;
          bgr[1] = 0;
          bgr[2] = 0;
        } else {
          const int x_color = std::get<0>(*xy_color);
          const int y_color = std::get<1>(*xy_color);
          bgr = color.at<cv::Vec3b>(y_color, x_color);
        }
      }
    }
  }
}

/////////////////
// Kinect2Impl //
/////////////////

Kinect2::Kinect2Impl::~Kinect2Impl() {
  if (dev_ == nullptr) return;
  Stop();
  dev_->close();
}

bool Kinect2::Kinect2Impl::Connect(const std::string& serial) {
  // Find device.
  if (serial.empty()) {
    dev_ = freenect_.openDefaultDevice();
  } else {
    dev_ = freenect_.openDevice(serial);
  }

  if (dev_ == nullptr) {
    return false;
  }

  // Set up frame listeners.
  dev_->setColorFrameListener(&listener_);
  dev_->setIrAndDepthFrameListener(&listener_);
  return true;
}

void Kinect2::Kinect2Impl::Start(bool color, bool depth) {
  is_running_ = true;
  dev_->startStreams(color, depth);
}

void Kinect2::Kinect2Impl::Stop() {
  is_running_ = false;
  dev_->stop();
  listener_.sem_color_callback.release();
  listener_.sem_depth_callback.release();
  if (thread_color_callback_.joinable()) {
    thread_color_callback_.join();
  }
  if (thread_depth_callback_.joinable()) {
    thread_depth_callback_.join();
  }
}

void Kinect2::Kinect2Impl::SetColorCallback(
    std::function<void(cv::Mat)>&& callback) {
  thread_color_callback_ =
      std::thread([Callback = std::move(callback), this]() {
        while (true) {
          // Wait for listener to receive a color frame.
          listener_.sem_color_callback.acquire();

          if (!is_running_) break;
          Callback(color_image());
        }
      });
}

void Kinect2::Kinect2Impl::SetDepthCallback(
    std::function<void(cv::Mat)>&& callback) {
  thread_depth_callback_ =
      std::thread([Callback = std::move(callback), this]() {
        while (true) {
          // Wait for listener to receive a depth frame.
          listener_.sem_depth_callback.acquire();

          if (!is_running_) break;
          Callback(depth_image());
        }
      });
}

cv::Mat Kinect2::Kinect2Impl::color_image() {
  listener_.GetBgrxFrame(img_bgrx_);
  cv::cvtColor(img_bgrx_, img_bgr_, cv::COLOR_BGRA2BGR);

  // Return a Mat wrapper without copying data.
  return img_bgrx_;
}

cv::Mat Kinect2::Kinect2Impl::depth_image() {
  listener_.GetDepthFrame(img_depth_);

  // Return a Mat wrapper without copying data.
  return img_depth_;
}

///////////////////
// FrameListener //
///////////////////

bool Kinect2::Kinect2Impl::FrameListener::onNewFrame(
    libfreenect2::Frame::Type type, libfreenect2::Frame* frame) {
  switch (type) {
    case libfreenect2::Frame::Type::Color:
      LockCopy(buffer_bgrx_.data(), frame->data, kColorBytes, mtx_bgrx_);
      sem_color_callback.release();
      break;
    case libfreenect2::Frame::Type::Depth:
      LockCopy(buffer_depth_.data(), frame->data, kDepthBytes, mtx_depth_);
      sem_depth_callback.release();
      break;
    default:
      break;
  }

  // Return false to let libfreenect2 reuse the frame.
  return false;
}

void Kinect2::Kinect2Impl::FrameListener::GetBgrxFrame(cv::Mat& img) {
  if (img.elemSize() != kColorBytes) {
    img = cv::Mat(kColorHeight, kColorWidth, CV_8UC4);
  }
  LockCopy(img.data, buffer_bgrx_.data(), kColorBytes, mtx_bgrx_);
};

void Kinect2::Kinect2Impl::FrameListener::GetDepthFrame(cv::Mat& img) {
  if (img.elemSize() != kDepthBytes) {
    img = cv::Mat(kDepthHeight, kDepthWidth, CV_32FC1);
  }
  LockCopy(img.data, buffer_depth_.data(), kDepthBytes, mtx_depth_);
};

void Kinect2::Kinect2Impl::FrameListener::LockCopy(void* dest, const void* src,
                                                   size_t bytes,
                                                   std::mutex& mtx) {
  std::lock_guard<std::mutex> lock(mtx);
  std::memcpy(dest, src, bytes);
}

// const auto depth_params = dev_->getIrCameraParams();
// const auto color_params = dev_->getColorCameraParams();
// listener_.registration_ =
//     std::make_unique<libfreenect2::Registration>(depth_params,
//     color_params);

// // Get depth parameters
// depth_intrinsic_matrix_ << depth_params.fx, 0., depth_params.cx, 0.,
//     depth_params.fy, depth_params.cy, 0., 0., 1.;
// depth_distortion_coeffs_ << depth_params.k1, depth_params.k2,
// depth_params.p1,
//     depth_params.p2, depth_params.k3;

// // Get color parameters
// rgb_intrinsic_matrix_ << color_params.fx, 0., color_params.cx, 0.,
//     color_params.fy, color_params.cy, 0., 0., 1.;
// // rgb_distortion_coeffs_.setZero();

// cv::Mat depth_distortion =
//     cv::Mat(1, 5, CV_64F, depth_distortion_coeffs_.data());
// cv::Mat depth_intrinsic =
//     cv::Mat(3, 3, CV_64F, depth_intrinsic_matrix_.data());
// cv::initUndistortRectifyMap(depth_intrinsic, depth_distortion, cv::Mat(),
//                             depth_intrinsic,
//                             cv::Size(kWidthDepth, kHeightDepth), CV_32FC1,
//                             distortion_maps_.first,
//                             distortion_maps_.second);

// bool Kinect2::FrameListener::onNewFrame(libfreenect2::Frame::Type type,
//                                         libfreenect2::Frame* frame) {
//   switch (type) {
//     case libfreenect2::Frame::Type::Color: {
//       std::lock_guard<std::mutex> lock(mtx_bgrx_);
//       std::memcpy(buffer_bgrx_.data(), frame->data, sizeof(buffer_bgrx_));
//     } break;
//     case libfreenect2::Frame::Type::Depth: {
//       std::lock_guard<std::mutex> lock(mtx_depth_);
//       std::memcpy(buffer_depth_.data(), frame->data, sizeof(buffer_depth_));
//     } break;
//     default:
//       break;
//   }
//   // if (type == libfreenect2::Frame::Type::Color) {
//   //   const size_t num_bytes = frame_color_->width * frame_color_->height *
//   //                            frame_color_->bytes_per_pixel;
//   //   std::memcpy(frame_color_->data, frame->data, num_bytes);
//   //   rgb_callback_(reinterpret_cast<const uint32_t*>(frame->data));
//   // } else if (type == libfreenect2::Frame::Type::Depth) {
//   //   // registration_->undistortDepth(frame,
//   //   frame_depth_undistorted_.get()); cv::Mat img_depth(kHeightDepth,
//   //   kWidthDepth, CV_32FC1, frame->data); cv::medianBlur(img_depth,
//   //   img_depth, 5); registration_->apply(
//   //       frame_color_.get(), frame, frame_depth_undistorted_.get(),
//   //       frame_color_registered_.get(), true, frame_depth_big_.get());
//   //   depth_callback_(reinterpret_cast<const float*>(frame->data));
//   // }

//   // Return false to let libfreenect2 manage frame memory.
//   return false;
// }

// Kinect2::Kinect2() {
//   listener_.frame_color_ =
//       std::make_unique<libfreenect2::Frame>(rgb_width(), rgb_height(), 4);
//   listener_.frame_color_->format = libfreenect2::Frame::Format::BGRX;
//   listener_.frame_depth_undistorted_ =
//       std::make_unique<libfreenect2::Frame>(depth_width(), depth_height(),
//       4);
//   listener_.frame_depth_undistorted_->format =
//       libfreenect2::Frame::Format::Float;
//   listener_.frame_color_registered_ =
//       std::make_unique<libfreenect2::Frame>(depth_width(), depth_height(),
//       4);
//   listener_.frame_color_registered_->format =
//   libfreenect2::Frame::Format::BGRX; listener_.frame_depth_big_ =
//       std::make_unique<libfreenect2::Frame>(rgb_width(), rgb_height() + 2,
//       4);
//   listener_.frame_depth_big_->format = libfreenect2::Frame::Format::Float;

//   listener_.rgb_callback_ = [this](const uint32_t* buffer) {
//     RgbCallback(buffer);
//   };
//   listener_.depth_callback_ = [this](const float* buffer) {
//     DepthCallback(buffer);
//   };

//   // Pulled from iai_kinect in ROS
//   depth_intrinsic_matrix_ << 3.6816072569315202e+02,
//   0., 2.4594416950015150e+02,
//       0., 3.6774098375548385e+02, 2.0405583803178158e+02, 0., 0., 1.;
//   depth_distortion_coeffs_ << 1.2784611385723821e-01,
//   -3.5757964278106197e-01,
//       -1.1644394534547308e-03,
//       -2.6300044551231176e-03, 1.6759244215113878e-01;

//   // Calibrated 09/04/19
//   rgb_intrinsic_matrix_ << 1.04557812e+03, 0., 9.58597924e+02, 0.,
//       1.04691518e+03, 5.40586651e+02, 0., 0., 1.;
//   rgb_distortion_coeffs_ << 0.03519172, 0.00169421, 0.00131543, -0.0004966,
//   -0.06287118;
// }

// void Kinect2::RgbCallback(const uint32_t* buffer) {
//   std::lock_guard<std::mutex> lock(mtx_rgb_);
//   if (!promise_rgb_) return;

//   // Copy data
//   for (size_t y = 0; y < kHeightRgb; y++) {
//     for (size_t x = 0; x < kWidthRgb; x++) {
//       // Flip about vertical axis
//       const size_t i = y * kWidthRgb + x;
//       const size_t i_flipped = y * kWidthRgb + (kWidthRgb - 1) - x;
//       const uint8_t* buffer_i = reinterpret_cast<const uint8_t*>(buffer + i);
//       uint8_t* buffer_rgb_i = buffer_rgb_.data() + 3 * i_flipped;
//       std::memcpy(buffer_rgb_i, buffer_i, 3);
//     }
//   }

//   // Set future
//   promise_rgb_->set_value(
//       cv::Mat(kHeightRgb, kWidthRgb, CV_8UC3, buffer_rgb_.data()));
//   promise_rgb_.reset();
// }

// void Kinect2::DepthCallback(const float* buffer) {
//   std::lock_guard<std::mutex> lock(mtx_depth_);
//   if (!promise_depth_) return;

//   // cv::Mat depth_undistorted = cv::Mat(kHeightDepth, kWidthDepth, CV_32FC1,
//   // reinterpret_cast<float*>(listener_.frame_depth_undistorted_->data));
//   // cv::Mat depth_raw = cv::Mat(kHeightDepth, kWidthDepth, CV_32FC1,
//   // const_cast<float*>(buffer)); cv::Mat depth_registered =
//   // cv::Mat(kHeightDepth, kWidthDepth, CV_32FC1, buffer_depth_.data());
//   // cv::flip(depth_undistorted, depth_registered, 1);
//   // depth_registered *= 0.001;

//   cv::Mat depth_big_raw(
//       rgb_height(), rgb_width(), CV_32FC1,
//       listener_.frame_depth_big_->data + sizeof(float) * rgb_width());
//   cv::Mat depth_big_registered(rgb_height(), rgb_width(), CV_32FC1,
//                                buffer_depth_big_.data());
//   for (size_t y = 0; y < kHeightRgb; y++) {
//     for (size_t x = 0; x < kWidthRgb; x++) {
//       const float d_raw = depth_big_raw.at<float>(y, x);
//       float& d_out = depth_big_registered.at<float>(y, kWidthRgb - 1 - x);
//       d_out = (d_raw >= 2000. || d_raw <= 500. || std::isnan(d_raw))
//                   ? std::nan("")
//                   : 0.001 * d_raw;
//     }
//   }
//   // cv::flip(depth_big_raw, depth_big_registered, 1);
//   // depth_big_registered *= 0.001;
//   // static cv::Mat depth_flipped = cv::Mat(kHeightDepth, kWidthDepth,
//   // CV_32FC1);

//   // cv::Mat depth_raw = cv::Mat(kHeightDepth, kWidthDepth, CV_32FC1,
//   // const_cast<float*>(buffer)); cv::flip(depth_raw, depth_flipped, 1);
//   // // depth_flipped -= 2.0119255597912145;

//   // cv::Mat depth_registered = cv::Mat(kHeightDepth, kWidthDepth, CV_32FC1,
//   // buffer_depth_.data()); cv::remap(depth_flipped, depth_registered,
//   // distortion_maps_.first, distortion_maps_.second, cv::INTER_NEAREST);
//   // depth_registered *= 0.001;

//   // Set future
//   // TODO: Fix race condition where user accesses buffer_rgb_ while this
//   // callback writes to it promise_depth_->set_value(depth_registered);
//   promise_depth_->set_value(depth_big_registered);
//   promise_depth_.reset();
// }

// cv::Mat Kinect2::depth_big() {
//   return cv::Mat(rgb_height() / 2, rgb_width() / 2, CV_32FC1,
//                  buffer_depth_big_.data());
// }

}  // namespace redis_rgbd
