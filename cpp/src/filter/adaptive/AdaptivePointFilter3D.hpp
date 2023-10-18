#pragma once
#include <Eigen/Dense>
#include <tuple>
#include <unsupported/Eigen/MatrixFunctions>

#include "../Point.hpp"

using Eigen::MatrixXd;

template <typename Value, typename AdaptiveFilter>
class AdaptivePointFilter3D {
    AdaptiveFilter x_filter;
    AdaptiveFilter y_filter;
    AdaptiveFilter z_filter;

public:
    static AdaptivePointFilter3D<Value, AdaptiveFilter> default_init(int joint, MatrixXd measurement_errors)
    {
        Point<Value> measurement_error;
        measurement_error.x = measurement_errors(joint, 0);
        measurement_error.y = measurement_errors(joint, 1);
        measurement_error.z = measurement_errors(joint, 2);

        return AdaptivePointFilter3D<Value, AdaptiveFilter>(measurement_error);
    }

    AdaptivePointFilter3D(Point<Value> measurement_error)
    {
        x_filter = AdaptiveFilter::default_init(measurement_error.x);
        y_filter = AdaptiveFilter::default_init(measurement_error.y);
        z_filter = AdaptiveFilter::default_init(measurement_error.z);
    }

    void init(Point<Value> initial_point)
    {
        auto initial_errors = MatrixXd(2, 2);
        initial_errors(0, 0) = 1;
        initial_errors(0, 1) = 0;
        initial_errors(1, 0) = 0;
        initial_errors(1, 1) = 1;

        auto initial_state_x = MatrixXd(2, 1);
        initial_state_x(0, 0) = initial_point.x;
        initial_state_x(1, 0) = 0;
        x_filter.init(initial_state_x, initial_errors);

        auto initial_state_y = MatrixXd(2, 1);
        initial_state_y(0, 0) = initial_point.y;
        initial_state_y(1, 0) = 0;
        y_filter.init(initial_state_y, initial_errors);

        auto initial_state_z = MatrixXd(2, 1);
        initial_state_z(0, 0) = initial_point.z;
        initial_state_z(1, 0) = 0;
        z_filter.init(initial_state_z, initial_errors);
    }

    std::tuple<Point<Value>, Point<Value>> step(Point<Value> value, Value time_diff)
    {
        Point<Value> position;
        Point<Value> velocity;
        std::tie(position.x, velocity.x) = x_filter.step(value.x, time_diff);
        std::tie(position.y, velocity.y) = y_filter.step(value.y, time_diff);
        std::tie(position.z, velocity.z) = z_filter.step(value.z, time_diff);
        return std::make_tuple(position, velocity);
    }
    // Consider a point an outlier if any of the axis are outliers
    // How to include confidence level
};
