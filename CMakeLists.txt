cmake_minimum_required(VERSION 3.20)
project(mlxpipe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(MLXPIPE_BUILD_EXAMPLES "Build the webcam examples" ON)
option(MLXPIPE_BUILD_TESTS "Build the tests" ON)

find_package(MLX CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc geometry)

# --- The library -------------------------------------------------------------
file(GLOB_RECURSE MLXPIPE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(FILTER MLXPIPE_SOURCES EXCLUDE REGEX "/src/c_api/")
add_library(mlxpipe ${MLXPIPE_SOURCES})
target_include_directories(mlxpipe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(mlxpipe PUBLIC mlx opencv_core opencv_imgproc opencv_geometry)
target_compile_options(mlxpipe PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-copy)

add_library(mlxpipe::mlxpipe ALIAS mlxpipe)

# The face C API as a standalone dylib: include/mlxpipe_face.h is its only
# header (no MLX or OpenCV headers needed). Only mlxpipe_* symbols are exported.
add_library(mlxpipe_face SHARED src/c_api/face.cpp)
target_link_libraries(mlxpipe_face PRIVATE mlxpipe)
target_compile_options(mlxpipe_face PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-copy)
target_link_options(mlxpipe_face PRIVATE "LINKER:-exported_symbol,_mlxpipe_*" "LINKER:-dead_strip")
set_target_properties(mlxpipe_face PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/mlxpipe_face.h
    INSTALL_NAME_DIR "@rpath")
add_library(mlxpipe::mlxpipe_face ALIAS mlxpipe_face)

# Installed as a CMake package: find_package(mlxpipe) -> mlxpipe::mlxpipe.
include(CMakePackageConfigHelpers)
install(TARGETS mlxpipe mlxpipe_face EXPORT mlxpipeTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
install(DIRECTORY include/mlxpipe DESTINATION include)
install(EXPORT mlxpipeTargets NAMESPACE mlxpipe:: DESTINATION lib/cmake/mlxpipe)
configure_package_config_file(cmake/mlxpipeConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/mlxpipeConfig.cmake
                              INSTALL_DESTINATION lib/cmake/mlxpipe)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/mlxpipeConfigVersion.cmake
                                 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mlxpipeConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/mlxpipeConfigVersion.cmake
        DESTINATION lib/cmake/mlxpipe)

# --- Examples ----------------------------------------------------------------
if(MLXPIPE_BUILD_EXAMPLES)
    find_package(OpenCV REQUIRED COMPONENTS videoio highgui imgcodecs)
    file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
    foreach(source ${EXAMPLE_SOURCES})
        get_filename_component(name ${source} NAME_WE)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE mlxpipe opencv_videoio opencv_highgui opencv_imgcodecs)
        target_compile_definitions(${name} PRIVATE MLXPIPE_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/models")
    endforeach()
endif()

# --- Tools -------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS imgcodecs)
add_executable(mlxpipe_replay tools/replay/replay.cpp)
target_link_libraries(mlxpipe_replay PRIVATE mlxpipe opencv_imgcodecs)
target_compile_definitions(mlxpipe_replay PRIVATE MLXPIPE_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/models")

# --- Tests -------------------------------------------------------------------
if(MLXPIPE_BUILD_TESTS)
    enable_testing()
    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*_test.cpp)
    foreach(source ${TEST_SOURCES})
        get_filename_component(name ${source} NAME_WE)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE mlxpipe)
        target_compile_definitions(${name} PRIVATE MLXPIPE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        add_test(NAME ${name} COMMAND ${name})
    endforeach()

    # The face C API, built as C against the dylib alone.
    enable_language(C)
    add_executable(c_api_test tests/c_api_test.c)
    target_link_libraries(c_api_test PRIVATE mlxpipe_face)
    target_include_directories(c_api_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(c_api_test PRIVATE MLXPIPE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(c_api_test PRIVATE -std=c99 -Wall -Wextra -pedantic)
    add_test(NAME c_api_test COMMAND c_api_test)

endif()
