#pragma once
#include <Eigen/Dense>
#include <tuple>
#include <unsupported/Eigen/MatrixFunctions>

#include "GenericFilter1D.hpp"
#include "Point.hpp"

std::tuple<MatrixXd, MatrixXd, MatrixXd> let_matricies_appear_magically()
{
    // Hard coded sub A = [1, Ts; 0, 1]
    MatrixXd A(2, 2);
    // A(0, 0) = 0;
    // A(0, 1) = 1;
    // A(1, 0) = 0;
    // A(1, 1) = 0;
    A(0, 0) = 1;
    A(0, 1) = 0;
    A(1, 0) = 0;
    A(1, 1) = 1;

    MatrixXd C(1, 2);
    C(0, 0) = 1;
    C(0, 1) = 0;

    MatrixXd G(2, 1);
    G(0, 0) = 0;
    G(1, 0) = 1;

    return { A, C, G };
}

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
        Value threshold = 2;
        measurement_noise.x = std::sqrt(measurement_errors(joint, 0)) * 10;
        measurement_noise.y = std::sqrt(measurement_errors(joint, 1)) * 10;
        measurement_noise.x = std::sqrt(measurement_errors(joint, 2)) * 10;

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
        std::tie(position.x, velocity.x) = x_filter.step(value.x, time_diff);
        std::tie(position.y, velocity.y) = y_filter.step(value.y, time_diff);
        std::tie(position.z, velocity.z) = z_filter.step(value.z, time_diff);
        return std::make_tuple(position, velocity);
    }
    // Consider a point an outlier if any of the axis are outliers
    // How to include confidence level
};
