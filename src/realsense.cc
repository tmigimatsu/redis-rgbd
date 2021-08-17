/**
 * realsense.cc
 *
 * Copyright 2019. All Rights Reserved.
 *
 * Created: July 29, 2019
 * Authors: Toki Migimatsu
 */

#include "redis_rgbd/realsense.h"

#include <algorithm>   // std::copy
#include <functional>  // std::function
#include <librealsense2/rs.hpp>

namespace {

static constexpr size_t kColorHeight = redis_rgbd::RealSense::kColorHeight;
static constexpr size_t kColorWidth = redis_rgbd::RealSense::kColorWidth;
static constexpr size_t kColorBytes = kColorHeight * kColorWidth * 3;

static constexpr size_t kDepthHeight = redis_rgbd::RealSense::kDepthHeight;
static constexpr size_t kDepthWidth = redis_rgbd::RealSense::kDepthWidth;
static constexpr size_t kDepthBytes =
    kDepthHeight * kDepthWidth * sizeof(float);
}  // namespace

namespace redis_rgbd {

const std::string RealSense::kName = "realsense";

class RealSense::RealSenseImpl {
 public:
  bool Connect(const std::string& serial);

  void Start(bool color, bool depth);

  void Stop();

  void ReceiveColor(const rs2::frame& frame);

  void ReceiveDepth(const rs2::frame& frame);

  void ListenFrames();

  std::vector<uint8_t>* GetColorFrame();
  std::vector<float>* GetDepthFrame();

  /**
   * Returns a container around locally-stored BGR image data.
   */
  cv::Mat color_image();

  /**
   * Returns a container around locally-stored depth image data.
   */
  cv::Mat depth_image();

  const cv::Mat& color_intrinsic_matrix() const {
    static const cv::Mat matrix(
        3, 3, CV_32FC1, const_cast<float*>(color_intrinsic_matrix_.data()));
    return matrix;
  };
  const std::array<float, 5>& color_distortion_coeffs() const {
    return color_distortion_coeffs_;
  }

  const cv::Mat& depth_intrinsic_matrix() const {
    static const cv::Mat matrix(
        3, 3, CV_32FC1, const_cast<float*>(depth_intrinsic_matrix_.data()));
    return matrix;
  };
  const std::array<float, 5>& depth_distortion_coeffs() const {
    return depth_distortion_coeffs_;
  }

 private:
  std::thread frame_thread_;
  std::atomic_bool runloop_;

  rs2::config cfg_;
  rs2::pipeline pipe_;

  std::mutex mtx_bgr_;
  std::vector<uint8_t> buffer_bgr_realsense_ =
      std::vector<uint8_t>(kColorBytes);
  bool is_bgr_updated_ = false;

  std::mutex mtx_depth_;
  std::vector<float> buffer_depth_realsense_ = std::vector<float>(kDepthBytes);
  bool is_depth_updated_ = false;

  std::vector<uint8_t> buffer_bgr_ = std::vector<uint8_t>(kColorBytes);
  std::vector<float> buffer_depth_ = std::vector<float>(kDepthBytes);

  std::array<float, 9> color_intrinsic_matrix_;
  std::array<float, 5> color_distortion_coeffs_;

  std::array<float, 9> depth_intrinsic_matrix_;
  std::array<float, 5> depth_distortion_coeffs_;

  float depth_scale_;
};

const cv::Mat& RealSense::color_intrinsic_matrix() const {
  return impl_->color_intrinsic_matrix();
};
const std::array<float, 5>& RealSense::color_distortion_coeffs() const {
  return impl_->color_distortion_coeffs();
}

const cv::Mat& RealSense::depth_intrinsic_matrix() const {
  return impl_->depth_intrinsic_matrix();
};
const std::array<float, 5>& RealSense::depth_distortion_coeffs() const {
  return impl_->depth_distortion_coeffs();
}

RealSense::RealSense() : impl_(std::make_unique<RealSenseImpl>()) {
  // Calibrated 09/04/19
  // rgb_intrinsic_matrix_ << 612.26415948, 0., 322.68317555, 0., 611.4546183,
  //     246.05608657, 0., 0., 1.;
  // rgb_distortion_coeffs_ << 4.89959428e-02, 4.01788409e-01, 6.12541686e-04,
  //     -2.14596079e-03, -1.83203863e+00;
}

RealSense::~RealSense() { Stop(); }

bool RealSense::Connect(const std::string& serial) {
  return impl_->Connect(serial);
}

bool RealSense::RealSenseImpl::Connect(const std::string& serial) {
  const rs2::context ctx;
  const rs2::device_list devices = ctx.query_devices();
  if (devices.size() == 0) return false;

  rs2::device dev;

  // Enable device
  if (serial.empty()) {
    dev = devices.front();
    cfg_.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
  } else {
    for (const rs2::device& it_dev : devices) {
      if (serial == it_dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
        dev = it_dev;
        break;
      }
    }
    if (!dev) return false;
    cfg_.enable_device(serial.c_str());
  }

  // Get depth scale
  rs2::depth_sensor sensor =
      dev.query_sensors().front().as<rs2::depth_sensor>();
  depth_scale_ = sensor.get_depth_scale();
  return true;
}

void RealSense::Start(bool color, bool depth) { impl_->Start(color, depth); }

void RealSense::RealSenseImpl::Start(bool color, bool depth) {
  cfg_.disable_all_streams();
  cfg_.enable_stream(RS2_STREAM_COLOR, -1, kColorWidth, kColorHeight,
                     RS2_FORMAT_BGR8);
  cfg_.enable_stream(RS2_STREAM_DEPTH, -1, kDepthWidth, kDepthHeight,
                     RS2_FORMAT_Z16);

  runloop_ = true;
  frame_thread_ = std::thread(&RealSense::RealSenseImpl::ListenFrames, this);

  const rs2::pipeline_profile profile = pipe_.start(cfg_);

  const rs2::video_stream_profile color_stream =
      profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
  const rs2_intrinsics color_intrinsics = color_stream.get_intrinsics();
  color_intrinsic_matrix_ = {color_intrinsics.fx,
                             0.,
                             color_intrinsics.ppx,
                             0.,
                             color_intrinsics.fy,
                             color_intrinsics.ppy,
                             0.,
                             0.,
                             1.};
  std::memcpy(color_distortion_coeffs_.data(), color_intrinsics.coeffs,
              sizeof color_intrinsics.coeffs);

  const rs2::video_stream_profile depth_stream =
      profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
  const rs2_intrinsics depth_intrinsics = depth_stream.get_intrinsics();
  depth_intrinsic_matrix_ = {depth_intrinsics.fx,
                             0.,
                             depth_intrinsics.ppx,
                             0.,
                             depth_intrinsics.fy,
                             depth_intrinsics.ppy,
                             0.,
                             0.,
                             1.};
  std::memcpy(depth_distortion_coeffs_.data(), depth_intrinsics.coeffs,
              sizeof depth_intrinsics.coeffs);
}

void RealSense::Stop() { impl_->Stop(); }

void RealSense::RealSenseImpl::Stop() {
  runloop_ = false;
  frame_thread_.join();
  pipe_.stop();
}

void RealSense::RealSenseImpl::ReceiveColor(const rs2::frame& frame) {
  std::lock_guard<std::mutex> lock(mtx_bgr_);
  const uint8_t* buffer = reinterpret_cast<const uint8_t*>(frame.get_data());
  std::memcpy(buffer_bgr_realsense_.data(), buffer, kColorBytes);
  is_bgr_updated_ = true;
}

void RealSense::RealSenseImpl::ReceiveDepth(const rs2::frame& frame) {
  std::lock_guard<std::mutex> lock(mtx_depth_);
  const uint8_t* buffer = reinterpret_cast<const uint8_t*>(frame.get_data());
  std::memcpy(buffer_depth_realsense_.data(), buffer, kDepthBytes);
  is_depth_updated_ = true;
}

void RealSense::RealSenseImpl::ListenFrames() {
  while (runloop_) {
    rs2::frameset frames = pipe_.wait_for_frames();
    ReceiveColor(frames.get_color_frame());
    ReceiveDepth(frames.get_depth_frame());
  }
}

std::vector<uint8_t>* RealSense::RealSenseImpl::GetColorFrame() {
  std::lock_guard<std::mutex> lock(mtx_bgr_);
  if (!is_bgr_updated_) return &buffer_bgr_;

  std::swap(buffer_bgr_, buffer_bgr_realsense_);
  is_bgr_updated_ = false;
  return &buffer_bgr_;
};

std::vector<float>* RealSense::RealSenseImpl::GetDepthFrame() {
  std::lock_guard<std::mutex> lock(mtx_depth_);
  if (!is_depth_updated_) return &buffer_depth_;

  std::swap(buffer_depth_, buffer_depth_realsense_);
  is_depth_updated_ = false;
  return &buffer_depth_;
};

cv::Mat RealSense::color_image() const { return impl_->color_image(); }

cv::Mat RealSense::RealSenseImpl::color_image() {
  std::vector<uint8_t>* buffer_bgr = GetColorFrame();
  return cv::Mat(kColorHeight, kColorWidth, CV_8UC3, buffer_bgr->data());
}

cv::Mat RealSense::depth_image() const { return impl_->depth_image(); }

cv::Mat RealSense::RealSenseImpl::depth_image() {
  std::vector<float>* buffer_depth = GetDepthFrame();
  return cv::Mat(kDepthHeight, kDepthWidth, CV_32FC1, buffer_depth->data());
}

}  // namespace redis_rgbd
