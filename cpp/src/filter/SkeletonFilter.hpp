#pragma once
#include <cstddef>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "Point.hpp"
#include "PointFilter3D.hpp"
#include "Utils.hpp"

using Eigen::MatrixXd;

template <typename Value>
class SkeletonFilter {
    size_t n_joints;
    bool initialized = false;
    Value last_time;

    std::vector<PointFilter3D<Value>> joint_filters;

public:
    int joint_count() { return n_joints; }

    SkeletonFilter(std::vector<Point<Value>> measurement_noises,
        std::vector<Point<Value>> system_noises, int m_n_joints,
        Value threshold)
        : n_joints(m_n_joints)
    {
        for (int i = 0; i < m_n_joints; ++i) {
            auto filter = PointFilter3D<Value>(measurement_noises[i],
                system_noises[i], threshold);
            joint_filters.push_back(filter);
        }
    }

    void init(std::vector<Point<Value>> inital_points, Value initial_time)
    {
        if (initialized) {
            return;
        }
        for (int i = 0; i < n_joints; ++i) {
            joint_filters[i].init(inital_points[i]);
        }
        last_time = initial_time;
        initialized = true;
    }

    bool is_initialized() { return initialized; }

    std::tuple<std::vector<Point<Value>>, std::vector<Point<Value>>> step(std::vector<Point<Value>> values,
        Value new_time)
    {
        std::vector<Point<Value>> positions;
        std::vector<Point<Value>> velocities;
        auto time_diff = new_time - last_time;
        // FIXME: Not nice using a 0..n_joints loop and push_back at the same time
        for (int i = 0; i < n_joints; ++i) {
            auto [position, velocity] = joint_filters[i].step(values[i], time_diff);
            positions.push_back(position);
            velocities.push_back(velocity);
        }
        return std::make_tuple(positions, velocities);
    }

    Point<Value> calculate_com(
        std::vector<Point<Value>> filtered_positions,
        MatrixXd MM) // MM[1x32]
    {
        Point<Value> com(0.0, 0.0, 0.0);
        for (int joint = 0; joint < this->joint_count(); ++joint) {
            com.x += filtered_positions[joint].x * MM(0, joint);
            com.y += filtered_positions[joint].y * MM(0, joint);
            com.z += filtered_positions[joint].z * MM(0, joint);
        }
        return com;
    }

    Point<Value> calculate_x_com(
        Point<Value> com,
        Point<Value> com_dot,
        Value l // length of inverted pendelum
    )
    {
        Value g = 9.81; // m/s
        Value w_0 = g / l;
        Point<Value> x_com(0.0, 0.0, 0.0);
        x_com.x = com.x + (com_dot.x / w_0);
        x_com.y = com.y + (com_dot.y / w_0);
        x_com.z = com.z + (com_dot.z / w_0);
        return x_com;
    }
};

template <typename Value>
class SkeletonFilterBuilder {
    std::string noise_data_path;
    int joint_count;
    std::vector<Point<Value>> measurement_noises;
    std::vector<Point<Value>> system_noises;
    Value threshold;

public:
    SkeletonFilterBuilder(std::string m_noise_data_path, int m_joint_count,
        Value m_threshold)
        : joint_count(m_joint_count)
    {
        noise_data_path = m_noise_data_path;
        threshold = m_threshold;

        auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(m_noise_data_path, m_joint_count);
        auto var = get_measurement_error(var_joints, m_joint_count, 209, 339);
        auto sqrt_var = var.array().sqrt();
        auto measurement_noise_for_all_joints = 10 * sqrt_var;

        Value factor_system_noise = 1.0 / 3;
        Value vmax = 10.0 * factor_system_noise;
        Value sigma_system_noise = vmax / 3;
        Value system_noise_x = std::pow(sigma_system_noise, 2);

        std::vector<Point<Value>> m_measurement_noises;
        std::vector<Point<Value>> m_system_noises;
        for (int joint = 0; joint < joint_count; ++joint) {
            m_measurement_noises.push_back(
                Point<Value>(measurement_noise_for_all_joints(joint, 0),
                    measurement_noise_for_all_joints(joint, 1),
                    measurement_noise_for_all_joints(joint, 2)));
            m_system_noises.push_back(
                Point<Value>(system_noise_x, system_noise_x, system_noise_x));
        }

        measurement_noises = m_measurement_noises;
        system_noises = m_system_noises;
    }

    SkeletonFilter<Value> build()
    {
        return SkeletonFilter<Value>(measurement_noises, system_noises, joint_count,
            threshold);
    }
};
