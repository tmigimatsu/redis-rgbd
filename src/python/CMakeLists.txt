############################################################
# CMakeLists for Pybind11
#
# Copyright 2021. All Rights Reserved.
#
# Created: May 24, 2021
# Authors: Toki Migimatsu
############################################################

project(pyredisrgbd)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIB_SOURCE_DIR}/redisrgbd")

ctrl_utils_add_subdirectory(pybind11)
ctrl_utils_add_subdirectory(Eigen3)

pybind11_add_module(pyredisrgbd redisrgbd.cc)

target_link_libraries(pyredisrgbd
  PUBLIC
    pybind11::pybind11
    redis_rgbd::redis_rgbd
    Eigen3::Eigen
)

# Put binary for installation temporarily in build/src/python.
install(TARGETS pyredisrgbd
    LIBRARY DESTINATION "${PROJECT_BINARY_DIR}"
)
