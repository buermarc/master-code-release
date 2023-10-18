#pragma once

#include <Eigen/src/Core/Matrix.h>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

#include <string>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

using Eigen::MatrixXd;
using Eigen::Tensor;

std::tuple<Tensor<double, 3>, int, std::vector<double>, std::vector<bool>>
load_data(std::string path, int joint_counts, int max_frames = -1)
{
    /**
     * Load data and calculate measurement noise
     * Returns loaded joints, and variance
     */
    std::ifstream file(path);
    json data = json::parse(file);
    int n_frames = data["frames"].size();

    // Only loead until max_frames if sensible
    if (max_frames != -1 && max_frames <= n_frames) {
        n_frames = max_frames;
    }

    Tensor<double, 3> joints(n_frames, joint_counts, 3);
    std::vector<double> timestamps;

    auto is_null = std::vector<bool>(n_frames, false);

    for (int i = 0; i < n_frames; ++i) {
        timestamps.push_back((double)data["frames"][i]["timestamp_usec"] * 1e-6);

        if (data["frames"][i]["bodies"][0].is_null()) {
            is_null[i] = true;
            std::cout << "Did find null, continue." << std::endl;
            continue;
        }

        auto joint_positions = data["frames"][i]["bodies"][0]["joint_positions"];
        for (int j = 0; j < joint_counts; ++j) {
            joints(i, j, 0) = joint_positions[j][0];
            joints(i, j, 1) = joint_positions[j][1];
            joints(i, j, 2) = joint_positions[j][2];
        }
    }

    joints = joints / 1000.0;

    return { joints, n_frames, timestamps, is_null };
}
MatrixXd get_measurement_error(Tensor<double, 3> joints, int joint_counts,
    int frame_start, int frame_end)
{
    MatrixXd var(joint_counts, 3);
    Eigen::array<Eigen::Index, 3> offsets;
    Eigen::array<Eigen::Index, 3> extents;
    Tensor<double, 0> mean_t;
    Tensor<double, 0> sum_t;
    for (int i = 0; i < joint_counts; ++i) {
        for (int j = 0; j < 2; ++j) {
            offsets = { frame_start, i, j };
            extents = { (frame_end - frame_start) + 1, 1, 1 };
            mean_t = joints.slice(offsets, extents).mean();
            sum_t = (joints.slice(offsets, extents) - mean_t(0)).pow(2).sum();
            var(i, j) = sum_t(0) / (frame_end - frame_start);
        }
        int j = 2;
        offsets = { frame_start, i, j };
        extents = { (frame_end - frame_start) + 1, 1, 1 };
        mean_t = joints.slice(offsets, extents).mean();
        sum_t = (joints.slice(offsets, extents) - mean_t(0)).pow(2).sum();
        var(i, j) = (sum_t(0) / (frame_end - frame_start) / 10); // treat 'z' as less important
    }
    return var;
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

/*
 *
    if dim not in [2, 3, 4]:
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [ .5*dt**3,    dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [ .5*dt**3,    dt**2,       dt],
             [ .5*dt**2,       dt,        1]]
    else:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
             [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
             [(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]

    if order_by_dim:
        return block_diag(*[Q]*block_size) * var
    return order_by_derivative(array(Q), dim, block_size) * var
    >>> Q_discrete_white_noise(2, dt=0.1, var=1., block_size=3)
def Q_discrete_white_noise(dim, dt=1., var=1., block_size=1, order_by_dim=True):
 */
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
