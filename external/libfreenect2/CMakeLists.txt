############################################################
# CMakeLists for dbot-redis
#
# Copyright 2020. All Rights Reserved.
#
# Created: September 24, 2020
# Authors: Toki Migimatsu
############################################################

if(NOT TARGET openkinect::libfreenect2)
    init_git_submodule(libfreenect2.git)
    set(search_command "s/^ADD_SUBDIRECTORY(\$\{MY_DIR\}\\/doc)/# ADD_SUBDIRECTORY(\$\{MY_DIR\}\\/doc)/")
    if(APPLE)
        execute_process(
            COMMAND sed -i "" -e "${search_command}" libfreenect2.git/CMakeLists.txt
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    else()
        execute_process(
            COMMAND sed "${search_command}" -i libfreenect2.git/CMakeLists.txt
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
    add_subdirectory(libfreenect2.git EXCLUDE_FROM_ALL)

    set(frame_listener_impl libfreenect2.git/src/frame_listener_impl.cpp)
    execute_process(
        COMMAND grep "functional" ${frame_listener_impl}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE is_functional_missing
    )
    if(${is_functional_missing})
        set(search_command "s/#include <libfreenect2\\/threading.h>/#include <libfreenect2\\/threading.h>\\n#include <functional>/")
        execute_process(
            COMMAND sed -i "${search_command}" ${frame_listener_impl}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
    add_library(openkinect::libfreenect2 INTERFACE IMPORTED GLOBAL)

    target_include_directories(openkinect::libfreenect2 INTERFACE
        libfreenect2.git/include
        ${CMAKE_CURRENT_BINARY_DIR}/libfreenect2.git)

    target_link_libraries(openkinect::libfreenect2 INTERFACE
        freenect2)
endif()
