FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade
RUN apt-get install -y build-essential curl wget vim software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y gcc-13 g++-13 cpp-13

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-x --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9
RUN update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-9 90

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 --slave /usr/bin/g++ g++ /usr/bin/g++-13 --slave /usr/bin/gcov gcov /usr/bin/gcov-13 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-13 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-13
RUN update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-13 130


WORKDIR /tmp/project

RUN apt-get install -y clang python3-dev python3-numpy python3-matplotlib python3-pip
RUN python3 -m pip install cmake
RUN echo "set editing-mode vi\nset keymap vi-command" > /root/.inputrc

COPY . .
