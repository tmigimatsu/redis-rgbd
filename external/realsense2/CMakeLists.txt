############################################################
# CMakeLists for dbot-redis
#
# Copyright 2020. All Rights Reserved.
#
# Created: September 24, 2020
# Authors: Toki Migimatsu
############################################################

if(NOT TARGET realsense2::realsense2)
    init_git_submodule(librealsense.git)

    execute_process(
        COMMAND sed -i "s/add_custom_target(uninstall/add_custom_target(realsense2_uninstall/" librealsense.git/CMake/install_config.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    # Disable realsense examples. This flag is an option that must be set globally.
    option(BUILD_EXAMPLES "Build realsense examples." OFF)
    add_subdirectory(librealsense.git)

    add_library(realsense2::realsense2 INTERFACE IMPORTED GLOBAL)

    # target_include_directories(openkinect::libfreenect2 INTERFACE
    #     libfreenect2.git/include
    #     ${CMAKE_CURRENT_BINARY_DIR}/libfreenect2.git)

    target_link_libraries(realsense2::realsense2 INTERFACE
        realsense2)
endif()
