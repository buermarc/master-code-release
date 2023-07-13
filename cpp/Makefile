CPP_FILES := $(shell find ./src -name '*.cpp')
FILES := $(shell find ./src -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c')

all: include/eigen build/filter

EIG_VER=3.4.0
include/eigen:
	wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-${EIG_VER}.tar.bz2 -O include/eigen-${EIG_VER}.tar.bz2
	cd include && tar xvf eigen-${EIG_VER}.tar.bz2 && rm eigen-${EIG_VER}.tar.bz2
	mv include/eigen-${EIG_VER} include/eigen

build/filter: build $(CPP_FILES)
	clang++ -g -std=c++20 -I include -I include/eigen ${CPP_FILES} -o build/filter

build:
	mkdir build

format:
	echo ${FILES}
	clang-format -i --style=WebKit ${FILES}
