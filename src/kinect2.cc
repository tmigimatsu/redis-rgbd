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
#include <numeric>  // std::isfinite
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

static constexpr int kFilterWidthHalf = 2;
static constexpr int kFilterHeightHalf = 1;
static constexpr float kFilterTolerance = 0.01;

constexpr float ComputeW(float mx, float my,
                         const std::array<float, 10>& params) {
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
  const float depth_cx = std::get<2>(kDepthIntrinsicMatrix);
  const float depth_cy = std::get<5>(kDepthIntrinsicMatrix);

  const float mx = (x - depth_cx) * kDepthQ;
  const float my = (y - depth_cy) * kDepthQ;

  const float wx = ComputeW(mx, my, kColorExtrinsicX);
  const float wy = ComputeW(mx, my, kColorExtrinsicY);

  const float color_fx = std::get<0>(kColorIntrinsicMatrix);
  const float color_cy = std::get<5>(kColorIntrinsicMatrix);

  const float rx = (wx / (color_fx * kColorQ)) - (kShiftM / kShiftD);
  const float ry = (wy / kColorQ) + color_cy;

  return {rx, ry};
}

using DepthMap = std::array<std::array<Arr2f, kDepthWidth>, kDepthHeight>;
constexpr DepthMap ComputeDepthColorMap() {
  DepthMap depth_color_map = {0};
  for (size_t y = 0; y < kDepthHeight; y++) {
    for (size_t x = 0; x < kDepthWidth; x++) {
      depth_color_map[y][x] = DepthToColor(x, y);
    }
  }
  return depth_color_map;
}

static constexpr DepthMap kDepthColorMap = ComputeDepthColorMap();

static std::optional<std::array<int, 2>> DepthToColorIndices(int x_depth,
                                                             int y_depth,
                                                             float z) {
  if (z <= 0 || !std::isfinite(z)) return {};

  const float map_x = kDepthColorMap[y_depth][x_depth][0];
  const float map_y = kDepthColorMap[y_depth][x_depth][1];

  const int ry_int = map_y + 0.5;
  if (ry_int < 0 || ry_int >= kColorHeight) return {};

  const float fx = std::get<0>(kColorIntrinsicMatrix);
  const float cx = std::get<2>(kColorIntrinsicMatrix);

  const float rx = (map_x + (kShiftM / z)) * fx + cx;
  const int rx_int = rx + 0.5;
  assert(rx_int >= 0 && rx_int < kColorWidth);

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

void Kinect2::RegisterColorToDepth(const cv::Mat& color, const cv::Mat& depth,
                                   cv::Mat& color_out) {
  // Check output size.
  if (color_out.rows != kDepthHeight || color_out.cols != kDepthWidth ||
      color_out.type() != CV_8UC3) {
    std::stringstream ss;
    ss << "Kinect2::RegisterColorToDepth(): color_out must be preallocated "
          "with size ["
       << kDepthHeight << ", " << kDepthWidth << "] and type CV_8UC3."
       << std::endl;
    std::invalid_argument(ss.str());
  }

  // Iterate over depth image.
  for (int y = 0; y < kDepthHeight; y++) {
    for (int x = 0; x < kDepthWidth; x++) {
      // Get corresponding color coordinates.
      const std::optional<std::array<int, 2>> xy_color =
          DepthToColorIndices(x, y, depth.at<float>(y, x));

      // Set output color pixel.
      cv::Vec3b& bgr = color_out.at<cv::Vec3b>(y, x);
      if (!xy_color.has_value()) {
        std::fill(std::begin(bgr.val), std::end(bgr.val), 0);
      } else {
        const std::array<int, 2>& xy = *xy_color;
        bgr = color.at<cv::Vec3b>(xy[1], xy[0]);
      }
    }
  }
}

void Kinect2::RegisterDepthToColor(const cv::Mat& depth, const cv::Mat& color,
                                   cv::Mat& depth_out) {
  // Check output size.
  if (depth_out.rows != kColorHeight || depth_out.cols != kColorWidth ||
      depth_out.type() != CV_32FC1) {
    std::stringstream ss;
    ss << "Kinect2::RegisterDepthToColor(): depth_out must be preallocated "
          "with size ["
       << kColorHeight << ", " << kColorWidth << "] and type CV_32FC1."
       << std::endl;
    std::invalid_argument(ss.str());
  }

  // Clear output image.
  depth_out.setTo(0);

  // Iterate over depth image.
  for (int y = 0; y < kDepthHeight; y++) {
    for (int x = 0; x < kDepthWidth; x++) {
      // Get corresponding color coordinates.
      const float z = depth.at<float>(y, x);
      const std::optional<std::array<int, 2>> xy_color =
          DepthToColorIndices(x, y, z);

      if (!xy_color.has_value()) continue;

      // Min box filter over color image.
      for (int yy = -kFilterHeightHalf; yy <= kFilterHeightHalf; yy++) {
        const int y_color = (*xy_color)[1] + yy;
        if (y_color < 0 || y_color >= kColorHeight) continue;

        for (int xx = -kFilterWidthHalf; xx <= kFilterWidthHalf; xx++) {
          const int x_color = (*xy_color)[0] + xx;
          assert(x_color > 0 && x_color < kColorWidth);

          // Set output pixel value to min z in window.
          float& zz = depth_out.at<float>(y_color, x_color);
          if (z < zz || zz == 0) zz = z;
        }
      }
    }
  }
}

void Kinect2::FilterDepth(const cv::Mat& depth_reg, cv::Mat& depth) {
  // Iterate over depth image.
  for (int y = 0; y < kDepthHeight; y++) {
    for (int x = 0; x < kDepthWidth; x++) {
      // Get corresponding color coordinates.
      float& z = depth.at<float>(y, x);
      const std::optional<std::array<int, 2>> xy_color =
          DepthToColorIndices(x, y, z);

      if (!xy_color.has_value()) continue;

      // Filter out noise larger than tolerance.
      const std::array<int, 2>& xy = *xy_color;
      const float z_min = depth_reg.at<float>(xy[1], xy[0]);
      const float error = (z - z_min) / z;
      if (error <= kFilterTolerance) continue;

      z = 0;
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
  return img_bgr_;
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

}  // namespace redis_rgbd
