FILES := $(shell find ./src ./tests ./include/filter -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c')
_TEST_BINS := $(shell find ./tests -name '*test.cpp')
TEST_BINS := $(_TEST_BINS:.cpp=)

.PHONY: build
build:
	FILTER_MAIN=1 FILTER_TEST=1 cmake -S . -B build
	FILTER_MAIN=1 FILTER_TEST=1 cmake --build build

.PHONY: tests
tests:
.PHONY: test
test:
	echo ${TEST_BINS}
	export ASSETS_DIR=$(shell pwd)/tests/assets && FILTER_TEST=1 cmake --build build && cd build && $(_TEST_BINS:.cpp=;)

.PHONY: format
format:
	echo ${FILES}
	clang-format -i --style=WebKit ${FILES}

DOCKER_TAG := filter-dev:latest
.PHONY: docker-build
docker-build:
	docker build . -t ${DOCKER_TAG}

build-in-dockerfile: docker-build
	docker run -it -v $(shell pwd):/tmp/project ${DOCKER_TAG} make _build-in-dockerfile

_build-in-dockerfile:
	FILTER_MAIN=1 FILTER_TEST=1 cmake -S . -B build-in-dockerfile
	FILTER_MAIN=1 FILTER_TEST=1 cmake --build build-in-dockerfile

simulation:
	cd simulations/E3/ && python E2_human_sts_forward.py ../../data

filter-simulation: simulation
	FILTER_MAIN=1 cmake -S . -B build && cmake --build build && ./build/load 0.0005

plot-simulation: filter-simulation
	cd plot && python animate.py
