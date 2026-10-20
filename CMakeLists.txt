cmake_minimum_required(VERSION 3.14)
project(MediaPipeMLXPort CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR} include src /usr/local/include)
link_directories(/usr/local/lib)

# Source files
set(SOURCES
    src/vision.cpp
    src/tensor_utils.cpp
    src/ssd_anchors.cpp
    src/non_max_suppression.cpp
    src/tensors_to_detections.cpp
    src/one_euro_filter.cpp
)

# 1. Build the dynamic library
add_library(mlx_vision SHARED ${SOURCES})
target_link_libraries(mlx_vision ${OpenCV_LIBS} mlx)
set_target_properties(mlx_vision PROPERTIES FRAMEWORK FALSE)

# 2. Build the test application
add_executable(test_app main.cpp)
target_link_libraries(test_app mlx_vision ${OpenCV_LIBS} mlx)
