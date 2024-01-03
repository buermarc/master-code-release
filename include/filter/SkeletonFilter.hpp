#pragma once
#include <cstddef>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include <filter/Point.hpp>
#include <filter/PointFilter3D.hpp>
#include <filter/SkeletonSaver.hpp>
#include <filter/Utils.hpp>
#include <filter/com.hpp>

using Eigen::MatrixXd;

template <typename Value>
class SkeletonFilter : public SkeletonStabilityMetrics<Value>, public SkeletonSaver<Value> {
    size_t n_joints;
    bool initialized = false;
    Value last_time;

    std::vector<PointFilter3D<Value>> joint_filters;

public:
    int joint_count() { return n_joints; }
    bool is_initialized() { return initialized; }

    SkeletonFilter(std::vector<Point<Value>> measurement_noises,
        std::vector<Point<Value>> system_noises, int m_n_joints,
        Value threshold, MatrixXd MM)
        : n_joints(m_n_joints)
        , SkeletonStabilityMetrics<Value>(MM)
    {
        for (int i = 0; i < m_n_joints; ++i) {
            auto filter = PointFilter3D<Value>(measurement_noises[i],
                system_noises[i], threshold);
            joint_filters.push_back(filter);
        }
    }

    SkeletonFilter(
        int m_n_joints,
        MatrixXd measurement_errors,
        MatrixXd MM,
        int threshold = 10
        )
        : n_joints(m_n_joints)
        , SkeletonStabilityMetrics<Value>(MM)
    {
        double factor_system_noise = 1.0 / 3;
        double vmax = 10.0 * factor_system_noise;
        double sigma_system_noise = vmax / 3;
        double system_noise = std::pow(sigma_system_noise, 2);
        for (int i = 0; i < m_n_joints; ++i) {
            auto filter = PointFilter3D<Value>::default_init(i, measurement_errors);
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

    std::tuple<std::vector<Point<Value>>, std::vector<Point<Value>>> step(std::vector<Point<Value>> values, Value new_time)
    {
        if (!initialized) {
            init(values, new_time);
            return std::make_tuple(values, std::vector<Point<Value>>());
        }

        std::vector<Point<Value>> positions;
        std::vector<Point<Value>> velocities;
        auto time_diff = new_time - last_time;
        // FIXME: Not nice using a 0..n_joints loop and push_back at the same time
        for (int i = 0; i < n_joints; ++i) {
            auto [position, velocity] = joint_filters[i].step(values[i], time_diff);
            positions.push_back(position);
            velocities.push_back(velocity);
        }

        SkeletonStabilityMetrics<Value>::store_step(positions, velocities);

        if (this->saver_enabled()) {
            this->save_step(new_time, values, positions, velocities);
        }

        last_time = new_time;
        return std::make_tuple(positions, velocities);
    }
};

template <typename Value>
class SkeletonFilterBuilder {
    int joint_count;
    std::vector<Point<Value>> measurement_noises;
    std::vector<Point<Value>> system_noises;
    Value threshold;

public:
    SkeletonFilterBuilder(int m_joint_count,
        Value m_threshold)
        : joint_count(m_joint_count)
    {
        threshold = m_threshold;

        auto var = get_cached_measurement_error();
        // auto sqrt_var = var.array().sqrt();
        auto measurement_noise_for_all_joints = var;

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
            threshold, get_azure_kinect_com_matrix());
    }
};
