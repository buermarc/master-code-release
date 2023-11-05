enable_testing()

add_executable(
    hello_test
    hello_test.cpp
)

add_executable(
    constrained_skeleton_filter_test
    constrained_skeleton_filter_test.cpp
)

target_link_libraries(
  hello_test
  GTest::gtest_main
)


target_link_libraries(
  constrained_skeleton_filter_test
  GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(hello_test)
target_link_libraries(hello_test nlohmann_json::nlohmann_json filter)
target_include_directories(hello_test PRIVATE ${CMAKE_SOURCE_DIR}/src)

gtest_discover_tests(constrained_skeleton_filter_test)
target_link_libraries(constrained_skeleton_filter_test nlohmann_json::nlohmann_json filter)
target_include_directories(constrained_skeleton_filter_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
