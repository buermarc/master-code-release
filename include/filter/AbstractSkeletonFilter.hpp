#pragma once
#include "filter/SkeletonSaver.hpp"
#include <filter/Point.hpp>
#include <filter/com.hpp>
#include <vector>
template <typename Value>
class AbstractSkeletonFilter : public SkeletonStabilityMetrics<Value>, public SkeletonSaver<Value> {
    public:
    AbstractSkeletonFilter() : SkeletonStabilityMetrics<Value>() {};
    virtual bool is_initialized() = 0;
    virtual Value time_diff(Value timestamp) = 0;
    virtual void init(std::vector<Point<Value>> initial_points, Value initial_time) = 0;
    virtual std::tuple<std::vector<Point<Value>>, std::vector<Point<Value>>> step(std::vector<Point<Value>> values, Value new_time) = 0;
};

template <typename Value>
class AbstractSkeletonFilterBuilder {
    public:
    virtual std::shared_ptr<AbstractSkeletonFilter<Value>> build() = 0;
};
