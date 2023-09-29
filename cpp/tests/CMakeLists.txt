enable_testing()

add_executable(
    hello_test
    hello_test.cpp
)
target_link_libraries(
  hello_test
  GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(hello_test)
target_link_libraries(hello_test nlohmann_json::nlohmann_json filter)
target_include_directories(hello_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
