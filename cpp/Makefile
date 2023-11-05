FILES := $(shell find ./src ./tests -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c')
_TEST_BINS := $(shell find ./tests -name '*test.cpp')
TEST_BINS := $(_TEST_BINS:.cpp=)

build:
	cmake -S . -b build

tests:
test:
	echo ${TEST_BINS}
	cmake --build build && cd build && $(_TEST_BINS:.cpp='')

format:
	echo ${FILES}
	clang-format -i --style=WebKit ${FILES}
