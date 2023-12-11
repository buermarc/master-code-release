#pragma once
#include <Eigen/Dense>
#include <tuple>
#include <unsupported/Eigen/MatrixFunctions>

#include "GenericFilter1D.hpp"
#include "Point.hpp"

using Eigen::MatrixXd;

std::tuple<MatrixXd, MatrixXd, MatrixXd> let_matricies_appear_magically();

template <typename Value>
class PointFilter3D {
    GenericFilter1D<Value> x_filter;
    GenericFilter1D<Value> y_filter;
    GenericFilter1D<Value> z_filter;

public:
    static PointFilter3D<Value> default_init(int joint, MatrixXd measurement_errors)
    {
        Point<Value> measurement_noise;
        Point<Value> system_noise;
        Value threshold = 10;
        measurement_noise.x = measurement_errors(joint, 0);
        measurement_noise.y = measurement_errors(joint, 1);
        measurement_noise.z = measurement_errors(joint, 2);

        Value factor_system_noise = 1.0 / 3;
        Value vmax = 10.0 * factor_system_noise;
        Value sigma_system_noise = vmax / 3;
        system_noise.x = std::pow(sigma_system_noise, 2);
        system_noise.y = system_noise.x;
        system_noise.z = system_noise.x;
        return PointFilter3D<Value>(
            measurement_noise,
            system_noise,
            threshold);
    }

    PointFilter3D(Point<Value> measurement_noise, Point<Value> system_noise,
        Value threshold)
    {
        auto [ad, c, g] = let_matricies_appear_magically();
        x_filter = GenericFilter1D<Value>(ad,
            ad, // FIXME: we do not care
            c, g,
            g, // FIXME: we do not care
            measurement_noise.x, system_noise.x, threshold);
        y_filter = GenericFilter1D<Value>(ad,
            ad, // FIXME: we do not care
            c, g,
            g, // FIXME: we do not care
            measurement_noise.y, system_noise.y, threshold);
        z_filter = GenericFilter1D<Value>(ad,
            ad, // FIXME: we do not care
            c, g,
            g, // FIXME: we do not care
            measurement_noise.z, system_noise.z, threshold);
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
        std::cout << "x axis" << std::endl;
        std::tie(position.x, velocity.x) = x_filter.step(value.x, time_diff);
        std::cout << "y axis" << std::endl;
        std::tie(position.y, velocity.y) = y_filter.step(value.y, time_diff);
        std::cout << "z axis" << std::endl;
        std::tie(position.z, velocity.z) = z_filter.step(value.z, time_diff);
        return std::make_tuple(position, velocity);
    }
};
