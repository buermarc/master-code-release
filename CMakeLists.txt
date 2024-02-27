cmake_minimum_required(VERSION 3.14)
project(kalman_filter)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})

###
# --- Adding Dependencies
###
include(FetchContent)
include(ExternalProject)

### Google Test
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

### nlohmann_json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

### EIGEN
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

### Boost (for serialization)
set(BOOST_INCLUDE_LIBRARIES archive serialization)
set(BOOST_ENABLE_CMAKE ON)

include(FetchContent)
FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.83.0/boost-1.83.0.tar.xz
)
FetchContent_MakeAvailable(Boost)

FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG HEAD
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
FetchContent_MakeAvailable(cnpy)

###
# --- Defininig Library
###

include_directories(include)
add_library(
    filter
    SHARED
    src/filter/com.cpp
    src/filter/PointFilter3D.cpp
    src/filter/SimplePointFilter3D.cpp
    src/filter/Utils.cpp
    src/filter/ConstrainedSkeletonFilter.cpp
)

target_link_libraries(filter nlohmann_json::nlohmann_json Boost::serialization)
add_dependencies(filter eigen)

target_include_directories(filter PUBLIC include)
target_include_directories(filter SYSTEM INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>")

# IF(DEFINED ENV{FILTER_TEST})
#     add_subdirectory(tests)
# ENDIF()

IF(DEFINED ENV{FILTER_MAIN})
    add_executable(main src/filter.cpp)
    target_link_libraries (main filter ${PYTHON_LIBRARIES} cnpy ZLIB::ZLIB)
    add_dependencies(main eigen)

    find_package(ZLIB)
    add_executable(load src/loadnpy.cpp)
    target_link_libraries(load cnpy ZLIB::ZLIB filter)
    add_dependencies(load eigen)

    add_executable(read_tsv src/read_tsv.cpp)
    target_link_libraries(load cnpy ZLIB::ZLIB filter)
    add_dependencies(load eigen)
ENDIF()


add_executable(track src/track_constant_vel.cpp)
target_link_libraries(track filter ${PYTHON_LIBRARIES})
