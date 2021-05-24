############################################################
# CMakeLists for the symbolic library.
#
# Copyright 2020. All Rights Reserved.
#
# Created: May 7, 2020
# Authors: Toki Migimatsu
############################################################

set(LIB_NAME ${PROJECT_NAME})

# Create the library.
add_library(${LIB_NAME} SHARED)
add_library(${LIB_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)
set_target_properties(${LIB_NAME} PROPERTIES CXX_EXTENSIONS OFF)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-steps=21708800")
endif()

# # Remove doctest code if tests are disabled.
# if(NOT ${LIB_CMAKE_NAME}_BUILD_TESTING)
#     target_compile_definitions(${LIB_NAME} PRIVATE DOCTEST_CONFIG_DISABLE)
# endif()

# # Enable clang tidy checks.
# if(${LIB_CMAKE_NAME}_CLANG_TIDY)
#     target_enable_clang_tidy(${LIB_NAME})
# endif()

# Set sources.
target_sources(${LIB_NAME}
  PRIVATE
    kinect2.cc
)

# Set include directories.
set(LIB_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${LIB_NAME}
  PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${LIB_INCLUDE_DIR}>"
)

# Link library dependencies.
lib_add_subdirectory(ctrl_utils)
ctrl_utils_add_subdirectory(OpenCV)
lib_add_subdirectory(libfreenect2)
target_link_libraries(${LIB_NAME}
  PUBLIC
    OpenCV::OpenCV
  PRIVATE
    openkinect::libfreenect2
    ctrl_utils::ctrl_utils
)

# # Build python wrapper.
# if(${LIB_CMAKE_NAME}_BUILD_PYTHON)
#     add_subdirectory(python)
# endif()

# Install the library.
install(TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}Targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# Install the headers.
install(DIRECTORY "${LIB_INCLUDE_DIR}/${LIB_NAME}"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Create LibConfig.cmake.
set(LIB_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${LIB_NAME}")
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${LIB_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${LIB_NAME}Config.cmake"
    INSTALL_DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

# Create LibConfigVersion.cmake.
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${LIB_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Install LibConfig.cmake and LibConfigVersion.cmake.
install(FILES
    "${PROJECT_BINARY_DIR}/${LIB_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/${LIB_NAME}ConfigVersion.cmake"
    DESTINATION ${LIB_CONFIG_INSTALL_DIR}
)

# Export targets to LibTargets.cmake.
install(EXPORT ${LIB_NAME}Targets
    DESTINATION ${LIB_CONFIG_INSTALL_DIR}
    NAMESPACE ${LIB_NAME}::
)

# Export targets to LibTargets.cmake for the build tree.
export(TARGETS ${LIB_NAME}
    NAMESPACE ${LIB_NAME}::
    FILE ${LIB_NAME}Targets.cmake
)
