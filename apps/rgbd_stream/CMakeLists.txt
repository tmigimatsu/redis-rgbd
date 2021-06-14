############################################################
# CMakeLists for the redis-rgbd library.
#
# Copyright 2021. All Rights Reserved.
#
# Created: May 20, 2020
# Authors: Toki Migimatsu
############################################################

# message(STATUS "Configuring ${PROJECT_NAME} examples.")

add_executable(rgbd_stream main.cc)

target_compile_features(rgbd_stream PUBLIC cxx_std_17)
set_target_properties(rgbd_stream PROPERTIES CXX_EXTENSIONS OFF)

ctrl_utils_add_subdirectory(Eigen3)
target_link_libraries(rgbd_stream
  PRIVATE
    ctrl_utils::ctrl_utils
    cpp_redis::cpp_redis
    Eigen3::Eigen
    redis_rgbd::redis_rgbd
)

if(SYMBOLIC_CLANG_TIDY)
    target_enable_clang_tidy(rgbd_stream)
endif()
