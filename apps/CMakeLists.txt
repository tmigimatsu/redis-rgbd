############################################################
# CMakeLists for the redis-rgbd library.
#
# Copyright 2021. All Rights Reserved.
#
# Created: May 20, 2020
# Authors: Toki Migimatsu
############################################################

lib_add_subdirectory(ctrl_utils)
ctrl_utils_add_subdirectory(cpp_redis)

add_subdirectory(rgbd_calibrate)
add_subdirectory(rgbd_listen)
add_subdirectory(rgbd_stream)
