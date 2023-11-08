#include <filter/Utils.hpp>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

using Eigen::MatrixXd;
using Eigen::Tensor;

std::tuple<Tensor<double, 3>, int, std::vector<double>, std::vector<bool>>
load_data(std::string path, int joint_counts, int max_frames)
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

MatrixXd get_cached_measurement_error()
{
    auto joint_counts = 32;
    auto dim = 3;
    double buffer[] = { 1.77399e-06, 1.68736e-06, 2.11718e-06, 2.47874e-06, 2.33367e-06, 2.99541e-06, 6.9249e-06, 1.60645e-05, 1.09893e-05, 8.82163e-06, 2.57317e-06, 2.39028e-06, 2.29286e-06, 5.15587e-07, 4.14838e-06, 1.52637e-05, 1.44428e-05, 9.0977e-06, 1.68322e-06, 1.14567e-07, 1.22626e-07, 2.67744e-07, 1.8679e-06, 1.66142e-07, 1.12785e-07, 1.88709e-06, 2.84577e-06, 8.68894e-06, 5.43723e-06, 1.62492e-06, 5.71942e-06, 1.70656e-06, 6.93022e-07, 9.15834e-07, 1.08337e-06, 1.28531e-06, 1.34258e-06, 1.61126e-06, 1.19744e-06, 1.8826e-06, 2.9808e-06, 4.50613e-06, 9.05469e-06, 1.17422e-06, 9.98517e-07, 6.36217e-07, 7.5578e-07, 8.0348e-07, 1.12497e-05, 8.19525e-06, 7.58396e-07, 3.84225e-07, 1.06889e-06, 1.8059e-06, 7.7111e-07, 5.85476e-07, 1.2206e-06, 1.50405e-06, 1.37384e-06, 1.72195e-05, 9.38562e-06, 1.33561e-06, 1.35912e-05, 4.85787e-06, 2.25625e-06, 2.81386e-06, 3.73601e-06, 3.82117e-06, 3.73675e-06, 3.07268e-06, 1.80033e-06, 1.94514e-06, 3.46578e-06, 2.49043e-06, 2.28562e-06, 3.80967e-06, 3.71059e-06, 3.3972e-06, 2.91799e-06, 3.16357e-06, 4.94211e-06, 2.76514e-06, 2.6215e-06, 5.97159e-07, 2.67564e-07, 3.83324e-07, 1.99541e-06, 9.02339e-07, 2.45711e-07, 2.73343e-07, 3.98656e-06, 5.00919e-06, 6.00291e-06, 5.99656e-06, 5.69788e-06, 5.33685e-06 };
    Eigen::Map<MatrixXd> var(buffer, joint_counts, dim);
    return var;
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
