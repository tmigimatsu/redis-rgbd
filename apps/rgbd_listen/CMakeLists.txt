############################################################
# CMakeLists for the redis-rgbd library.
#
# Copyright 2021. All Rights Reserved.
#
# Created: May 24, 2020
# Authors: Toki Migimatsu
############################################################

add_executable(rgbd_listen main.cc)

target_compile_features(rgbd_listen PUBLIC cxx_std_17)
set_target_properties(rgbd_listen PROPERTIES CXX_EXTENSIONS OFF)

target_link_libraries(rgbd_listen
  PRIVATE
    ctrl_utils::ctrl_utils
    cpp_redis::cpp_redis
    redis_rgbd::redis_rgbd
)

if(SYMBOLIC_CLANG_TIDY)
    target_enable_clang_tidy(rgbd_listen)
endif()
