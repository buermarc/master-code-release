FILES := $(shell find ./src ./tests -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c')
_TEST_BINS := $(shell find ./tests -name '*test.cpp')
TEST_BINS := $(_TEST_BINS:.cpp=)

.PHONY: build
build:
	cmake -S . -b build

.PHONY: tests
tests:
.PHONY: test
test:
	echo ${TEST_BINS}
	cmake --build build && cd build && $(_TEST_BINS:.cpp='')

.PHONY: format
format:
	echo ${FILES}
	clang-format -i --style=WebKit ${FILES}

DOCKER_TAG := filter-dev:latest
.PHONY: docker-build
docker-build:
	docker build . -t ${DOCKER_TAG}
