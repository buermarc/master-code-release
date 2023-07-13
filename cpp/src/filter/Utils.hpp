#pragma once

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

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
