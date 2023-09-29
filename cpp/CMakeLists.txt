cmake_minimum_required(VERSION 3.14)
project(kalman_filter)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    json 
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

set(EIGEN_BUILD_DIR   ${CMAKE_BINARY_DIR}/eigen)
set(EIGEN_INSTALL_DIR ${CMAKE_SOURCE_DIR}/include/eigen3)
set(EIGEN_INCLUDE_DIR ${EIGEN_INSTALL_DIR})

ExternalProject_Add(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    PREFIX ${EIGEN_BUILD_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${EIGEN_BUILD_DIR}/src/eigen/Eigen ${EIGEN_INCLUDE_DIR}/Eigen 
      && ${CMAKE_COMMAND} -E copy_directory ${EIGEN_BUILD_DIR}/src/eigen/unsupported ${EIGEN_INCLUDE_DIR}/unsupported
    TEST_COMMAND ""
)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

add_subdirectory(tests)
add_subdirectory(src/filter)

add_executable (main src/filter.cpp)
target_link_libraries (main nlohmann_json::nlohmann_json filter)
target_include_directories (main PUBLIC "${PROJECT_SOURCE_DIR}/src")
add_dependencies (main eigen)
