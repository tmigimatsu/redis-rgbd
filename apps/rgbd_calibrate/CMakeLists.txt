############################################################
# CMakeLists for the redis-rgbd library.
#
# Copyright 2021. All Rights Reserved.
#
# Created: June 11, 2021
# Authors: Toki Migimatsu
############################################################

add_executable(rgbd_calibrate main.cc)

target_compile_features(rgbd_calibrate PUBLIC cxx_std_17)
set_target_properties(rgbd_calibrate PROPERTIES CXX_EXTENSIONS OFF)

ctrl_utils_add_subdirectory(Eigen3)
target_link_libraries(rgbd_calibrate
  PRIVATE
    ctrl_utils::ctrl_utils
    cpp_redis::cpp_redis
    Eigen3::Eigen
    redis_rgbd::redis_rgbd
)

if(SYMBOLIC_CLANG_TIDY)
    target_enable_clang_tidy(rgbd_calibrate)
endif()
