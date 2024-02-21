#pragma once
#include <cstddef>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include <filter/AbstractSkeletonFilter.hpp>
#include <filter/Point.hpp>
#include <filter/PointFilter3D.hpp>
#include <filter/SkeletonSaver.hpp>
#include <filter/Utils.hpp>
#include <filter/com.hpp>

using Eigen::MatrixXd;

template <typename Value>
class SkeletonFilter : public AbstractSkeletonFilter<Value> {
    size_t n_joints;
    bool initialized = false;
    Value last_time;

    std::vector<PointFilter3D<Value>> joint_filters;

public:
    int joint_count() { return n_joints; }
    bool is_initialized() override { return initialized; }

    SkeletonFilter(std::vector<Point<Value>> measurement_noises,
        std::vector<Point<Value>> system_noises, int m_n_joints, double factor,
        Value threshold, MatrixXd MM)
        : n_joints(m_n_joints)
        , AbstractSkeletonFilter<Value>()
    {
        this->set_filter_type("SkeletonFilter");
        this->set_measurement_error_factor(factor);

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
        int threshold = 10)
        : n_joints(m_n_joints)
        , AbstractSkeletonFilter<Value>()
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

    void init(std::vector<Point<Value>> initial_points, Value initial_time) override
    {
        if (initialized) {
            return;
        }
        for (int i = 0; i < n_joints; ++i) {
            joint_filters[i].init(initial_points[i]);
        }
        if (this->saver_enabled()) {
            std::vector<Point<Value>> velocities(initial_points.size());
            this->save_step(initial_time, initial_points, initial_points, velocities);
        }
        last_time = initial_time;
        initialized = true;
    }

    Value time_diff(Value new_time) override
    {
        if (!initialized)
            return 0;
        return new_time - last_time;
    }

    std::tuple<std::vector<Point<Value>>, std::vector<Point<Value>>> step(std::vector<Point<Value>> values, Value new_time) override
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

    std::tuple<std::vector<Point<Value>>, std::vector<Point<Value>>, std::vector<Point<Value>>> step_(std::vector<Point<Value>> values, Value new_time) override
    {
        if (!initialized) {
            init(values, new_time);
            return std::make_tuple(values, std::vector<Point<Value>>(), std::vector<Point<Value>>());
        }

        std::vector<Point<Value>> positions;
        std::vector<Point<Value>> velocities;
        std::vector<Point<Value>> predictions;
        auto time_diff = new_time - last_time;
        // FIXME: Not nice using a 0..n_joints loop and push_back at the same time
        for (int i = 0; i < n_joints; ++i) {
            auto [position, velocity, predicted] = joint_filters[i].step_(values[i], time_diff);
            positions.push_back(position);
            velocities.push_back(velocity);
            predictions.push_back(predicted);
        }

        SkeletonStabilityMetrics<Value>::store_step(positions, velocities);

        if (this->saver_enabled()) {
            this->save_step(new_time, values, positions, velocities, predictions);
        }

        last_time = new_time;
        return std::make_tuple(positions, velocities, predictions);
    }
};

template <typename Value>
class SkeletonFilterBuilder : public AbstractSkeletonFilterBuilder<Value> {
    int joint_count;
    std::vector<Point<Value>> measurement_noises;
    std::vector<Point<Value>> system_noises;
    Value threshold;
    double m_factor;

public:
    SkeletonFilterBuilder(int m_joint_count,
        Value m_threshold, double measurement_error_factor=5.0)
        : joint_count(m_joint_count)
    {
        threshold = m_threshold;

        auto var = get_cached_measurement_error(measurement_error_factor);
        m_factor = measurement_error_factor;
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

    std::shared_ptr<AbstractSkeletonFilter<Value>> build() override
    {
        return std::make_shared<SkeletonFilter<Value>>(measurement_noises, system_noises, joint_count, m_factor,
            threshold, get_azure_kinect_com_matrix());
    }
};
