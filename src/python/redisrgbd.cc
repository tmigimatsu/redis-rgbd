/**
 * symbolic.cc
 *
 * Copyright 2020. All Rights Reserved.
 *
 * Created: March 7, 2020
 * Authors: Toki Migimatsu
 */

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>  // std::invalid_argument
#include <sstream>    // std::stringstream

#include "redis_rgbd/kinect2.h"

namespace redis_rgbd {

namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT(google-build-using-namespace)

PYBIND11_MODULE(pyredisrgbd, m) {
  m.doc() = R"pbdoc(
    redis-rgbd Python API
    =====================

    Python wrapper for the redis-rgbd library.

    .. currentmodule:: redisrgbd
  )pbdoc";

  // Pddl
  py::class_<Kinect2>(m, "Kinect2")
      .def_property_readonly_static(
          "color_width", [](py::object) { return Kinect2::kColorWidth; },
          R"pbdoc(
            Color image width.
          )pbdoc")
      .def_property_readonly_static(
          "color_height", [](py::object) { return Kinect2::kColorHeight; },
          R"pbdoc(
            Color image height.
          )pbdoc")
      .def_property_readonly_static(
          "color_channel", [](py::object) { return Kinect2::kColorChannel; },
          R"pbdoc(
            Color image channel type.
          )pbdoc")
      .def_property_readonly_static(
          "depth_width", [](py::object) { return Kinect2::kDepthWidth; },
          R"pbdoc(
            Depth image width.
          )pbdoc")
      .def_property_readonly_static(
          "depth_height", [](py::object) { return Kinect2::kDepthHeight; },
          R"pbdoc(
            Depth image height.
          )pbdoc")
      .def_property_readonly_static(
          "depth_channel", [](py::object) { return Kinect2::kDepthChannel; },
          R"pbdoc(
            Depth image channel type.
          )pbdoc")
      .def_property_readonly_static(
          "color_intrinsic_matrix",
          [](py::object) {
            return Eigen::Map<
                const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
                Kinect2::kColorIntrinsicMatrix.data());
          },
          R"pbdoc(
            Color intrinsic matrix coefficients [fx, 0, cx; 0, fy, cy; 0, 0, 1].
          )pbdoc")
      .def_property_readonly_static(
          "color_distortion_coeffs",
          [](py::object) {
            return Eigen::Map<const Eigen::Matrix<float, 5, 1>>(
                Kinect2::kColorDistortionCoeffs.data());
          },
          R"pbdoc(
            Color distortion coefficients [k1, k2, p1, p2, k3].
          )pbdoc")
      .def_property_readonly_static(
          "depth_intrinsic_matrix",
          [](py::object) {
            Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>
                matrix(Kinect2::kDepthIntrinsicMatrix.data());
            return matrix;
          },
          R"pbdoc(
            Depth intrinsic matrix coefficients [fx, 0, cx; 0, fy, cy; 0, 0, 1].
          )pbdoc")
      .def_property_readonly_static(
          "depth_distortion_coeffs",
          [](py::object) {
            return Eigen::Map<const Eigen::Matrix<float, 5, 1>>(
                Kinect2::kDepthDistortionCoeffs.data());
          },
          R"pbdoc(
            Depth distortion coefficients [k1, k2, p1, p2, k3].
          )pbdoc")
      .def_static(
          "register_color_to_depth",
          [](py::array_t<uint8_t> color, py::array_t<float> depth,
             py::array_t<uint8_t> color_out) {
            py::buffer_info buf_color = color.request();
            py::buffer_info buf_depth = depth.request();
            py::buffer_info buf_color_out = color_out.request();

            if (buf_color.ndim != 3 ||
                buf_color.shape[0] != Kinect2::kColorHeight ||
                buf_color.shape[1] != Kinect2::kColorWidth ||
                buf_color.shape[2] != 3) {
              std::stringstream ss;
              ss << "color must have dimensions [" << Kinect2::kColorHeight
                 << ", " << Kinect2::kColorWidth << ", " << 3 << "].";
              throw std::runtime_error(ss.str());
            }

            if (buf_depth.ndim != 2 ||
                buf_depth.shape[0] != Kinect2::kDepthHeight ||
                buf_depth.shape[1] != Kinect2::kDepthWidth) {
              std::stringstream ss;
              ss << "depth must have dimensions [" << Kinect2::kDepthHeight
                 << ", " << Kinect2::kDepthWidth << "].";
              throw std::invalid_argument(ss.str());
            }

            if (buf_color_out.ndim != 3 ||
                buf_color_out.shape[0] != Kinect2::kDepthHeight ||
                buf_color_out.shape[1] != Kinect2::kDepthWidth ||
                buf_color_out.shape[2] != 3) {
              std::stringstream ss;
              ss << "color_out must have dimensions [" << Kinect2::kDepthHeight
                 << ", " << Kinect2::kDepthWidth << ", " << 3 << "].";
              throw std::invalid_argument(ss.str());
            }

            const cv::Mat img_color(Kinect2::kColorHeight, Kinect2::kColorWidth,
                                    CV_8UC3, buf_color.ptr);
            const cv::Mat img_depth(Kinect2::kDepthHeight, Kinect2::kDepthWidth,
                                    CV_32FC1, buf_depth.ptr);
            cv::Mat img_color_out(Kinect2::kDepthHeight, Kinect2::kDepthWidth,
                                  CV_8UC3, buf_color_out.ptr);

            Kinect2::RegisterColorToDepth(img_color, img_depth, img_color_out);
          },
          "color"_a, "depth"_a, "color_out"_a, R"pbdoc(
             Undistorts the depth image and registers the color image with it.

             Args:
                 color: 1920 x 1080 uint8 BGR image.
                 depth: 512 x 424 float32 depth image.
                 color_out: 512 x 424 uint8 registered BGR image.
            )pbdoc")
      .def_static(
          "register_depth_to_color",
          [](py::array_t<float> depth, py::array_t<float> depth_out) {
            py::buffer_info buf_depth = depth.request();
            py::buffer_info buf_depth_out = depth_out.request();

            if (buf_depth.ndim != 2 ||
                buf_depth.shape[0] != Kinect2::kDepthHeight ||
                buf_depth.shape[1] != Kinect2::kDepthWidth) {
              std::stringstream ss;
              ss << "depth must have dimensions [" << Kinect2::kDepthHeight
                 << ", " << Kinect2::kDepthWidth << "].";
              throw std::invalid_argument(ss.str());
            }

            if (buf_depth_out.ndim != 2 ||
                buf_depth_out.shape[0] != Kinect2::kColorHeight ||
                buf_depth_out.shape[1] != Kinect2::kColorWidth) {
              std::stringstream ss;
              ss << "depth_out must have dimensions [" << Kinect2::kColorHeight
                 << ", " << Kinect2::kColorWidth << ", " << 3 << "].";
              throw std::runtime_error(ss.str());
            }

            const cv::Mat img_depth(Kinect2::kDepthHeight, Kinect2::kDepthWidth,
                                    CV_32FC1, buf_depth.ptr);
            cv::Mat img_depth_out(Kinect2::kColorHeight, Kinect2::kColorWidth,
                                  CV_32FC1, buf_depth_out.ptr);

            Kinect2::RegisterDepthToColor(img_depth, img_depth_out);
          },
          "depth"_a, "depth_out"_a, R"pbdoc(
             Undistorts the depth image and registers the color image with it.

             Args:
                 depth: 512 x 424 float32 depth image.
                 depth_out: 1920 x 1080 uint8 registered depth image.
            )pbdoc")
      .def_static(
          "filter_depth",
          [](py::array_t<float> depth_reg, py::array_t<float> depth) {
            py::buffer_info buf_depth = depth.request();
            py::buffer_info buf_depth_reg = depth_reg.request();

            if (buf_depth_reg.ndim != 2 ||
                buf_depth_reg.shape[0] != Kinect2::kColorHeight ||
                buf_depth_reg.shape[1] != Kinect2::kColorWidth) {
              std::stringstream ss;
              ss << "depth_reg must have dimensions [" << Kinect2::kColorHeight
                 << ", " << Kinect2::kColorWidth << "].";
              throw std::runtime_error(ss.str());
            }

            if (buf_depth.ndim != 2 ||
                buf_depth.shape[0] != Kinect2::kDepthHeight ||
                buf_depth.shape[1] != Kinect2::kDepthWidth) {
              std::stringstream ss;
              ss << "depth must have dimensions [" << Kinect2::kDepthHeight
                 << ", " << Kinect2::kDepthWidth << "].";
              throw std::invalid_argument(ss.str());
            }

            const cv::Mat img_depth_reg(Kinect2::kColorHeight,
                                        Kinect2::kColorWidth, CV_32FC1,
                                        buf_depth_reg.ptr);

            cv::Mat img_depth(Kinect2::kDepthHeight, Kinect2::kDepthWidth,
                              CV_32FC1, buf_depth.ptr);

            Kinect2::FilterDepth(img_depth_reg, img_depth);
          },
          "depth_reg"_a, "depth"_a, R"pbdoc(
             Filters out noise in the depth image using a box filter.

             Args:
                 depth_reg: 1920 x 1080 float32 registered depth image computed
                            with `Kinect2.register_depth_to_color()`.
                 depth: 512 x 424 float32 depth image to be filtered in place.
            )pbdoc");

  py::add_ostream_redirect(m);
}

}  // namespace redis_rgbd
