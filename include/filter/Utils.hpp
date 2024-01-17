#pragma once
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

#include <string>

using Eigen::MatrixXd;
using Eigen::Tensor;

std::tuple<Tensor<double, 3>, int, std::vector<double>, std::vector<bool>>
load_filtered_data(std::string path, int joint_counts, int max_frames = -1);

std::tuple<Tensor<double, 3>, int, std::vector<double>, std::vector<bool>>
load_data(std::string path, int joint_counts, int max_frames = -1);

MatrixXd get_cached_measurement_error();

MatrixXd _get_measurement_error(Tensor<double, 3> joints, int joint_counts, int frame_start, int frame_end);

template <typename Value>
MatrixXd Q_discrete_white_noise_2d(Value time_diff, Value variation)
{
    MatrixXd system_noise(2, 2);
    system_noise(0, 0) = 0.25 * std::pow(time_diff, 4);
    system_noise(1, 0) = 0.5 * std::pow(time_diff, 3);
    system_noise(0, 1) = 0.5 * std::pow(time_diff, 3);
    system_noise(1, 1) = std::pow(time_diff, 2);
    return system_noise;
}

template <typename E, typename X>
void unroll(const std::vector<E>& v, std::set<X>& out)
{
    out.insert(v.begin(), v.end());
}

template <typename E, typename X>
void unroll(const std::vector<E>& v, std::vector<X>& out)
{
    out.insert(out.end(), v.begin(), v.end());
}

template <typename V, typename X>
void unroll(const std::vector<std::vector<V>>& v, std::vector<X>& out)
{
    for (const auto& e : v)
        unroll(e, out);
}

template <typename V, typename X>
void unroll(const std::vector<std::vector<V>>& v, std::set<X>& out)
{
    for (const auto& e : v)
        unroll(e, out);
}
