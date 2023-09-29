CPP_FILES := $(shell find ./src -name '*.cpp')
FILES := $(shell find ./src .tests -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c')

all: include/eigen build/filter

build:
	cmake -S . -b build

tests:
test:
	cmake --build build && ./build/tests/hello_test

format:
	echo ${FILES}
	clang-format -i --style=WebKit ${FILES}
