#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "filter/SkeletonFilter.hpp"
#include "filter/Utils.hpp"

using json = nlohmann::json;

using Eigen::MatrixXd;
using Eigen::Tensor;

int main()
{
    std::string var_path("../matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json");
    int joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);

    std::string data_path("../matlab/sts_NFOV_UNBINNED_720P_30fps.json");
    auto [joints, n_frames, timestamps, is_null] = load_data(data_path, joint_count, 870);

    if (std::find(is_null.begin(), is_null.end(), true) != is_null.end()) {
        std::cout << "found null" << std::endl;
    }

    std::cout << var << std::endl;
    std::cout << "This should only be printed once" << std::endl;

    double time_diff = 0.5; // some value

    MatrixXd A(2, 2);
    A(0, 0) = 0;
    A(0, 1) = 1;
    A(1, 0) = 0;
    A(1, 1) = 0;

    std::cout << A << std::endl;

    // will probably not work, but replace where A == 1 element with time_diff
    // simplify(expm(A * T_s)) but in c++
    // uses 0/1 array A to replace with T_s where 1, and A.exp() for expm(...)
    auto Ad = (A.array() * time_diff + (1 - A.array()) * A.exp().array()).matrix();
    std::cout << Ad << std::endl;

    MatrixXd G(2, 1);
    G(0, 0) = 0;
    G(1, 0) = 1;

    MatrixXd C(1, 2);
    C(0, 0) = 1;
    C(0, 1) = 0;
    std::cout << "C: " << C << std::endl;

    // seems to be the best approximation for matlab length(matrix)
    int d = std::fmax(A.rows(), A.cols());

    // Here is the integration step
    // Gd = simplify(int(expm(Ts*A)*G)); % fuer reale Abtastung
    // (Abtast-Halte-Glied) auto Gd = Ad * G; We have to hard code the integration
    // Where A = 1 => do integration, where 0 = 1
    // G = [T^2 / 2, T]
    G(0, 0) = std::pow(time_diff, 2) / 2;
    G(1, 0) = time_diff;

    // Measurement Noise
    // We have a measurement noise approx for each x, y, z for each joint
    // => var ; var.shape = [32, 3]

    // Systems Noise
    double factor_system_noise = 1.0 / 3;
    double vmax = 10.0 * factor_system_noise;
    double sigma_system_noise = vmax / 3;
    double system_noise_x = std::pow(sigma_system_noise, 2);
    double system_noise_y = std::pow(sigma_system_noise, 2);
    double system_noise_z = std::pow(sigma_system_noise, 2);
    std::cout << "system_noise_x :" << system_noise_x << std::endl;

    auto m_threshold = 2;

    auto sqrt_var = var.array().sqrt();
    auto measurement_noise_for_all_joints = 10 * sqrt_var;

    // Let's do it for all axis
    //
    std::vector<Point<double>> measurement_noises;
    std::vector<Point<double>> system_noises;
    std::vector<Point<double>> initial_points;
    for (int joint = 0; joint < joint_count; ++joint) {
        measurement_noises.push_back(
            Point<double>(measurement_noise_for_all_joints(joint, 0),
                measurement_noise_for_all_joints(joint, 1),
                measurement_noise_for_all_joints(joint, 2)));
        system_noises.push_back(
            Point<double>(system_noise_x, system_noise_x, system_noise_x));
        initial_points.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }
    SkeletonFilter<double> skeleton_filter(measurement_noises, system_noises, 32,
        m_threshold);
    skeleton_filter.init(initial_points, timestamps[0]);

    std::vector<std::vector<Point<double>>> filtered_values;
    // Add initial value
    std::vector<Point<double>> initial_joints;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_joints.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }
    filtered_values.push_back(initial_joints);

    int max_frame = n_frames;
    std::cout << "n_frames " << n_frames << std::endl;
    for (int frame_idx = 1; frame_idx < max_frame; ++frame_idx) {
        if (is_null[frame_idx])
            continue;
        // double time_diff = timestamps[frame_idx] - timestamps[frame_idx-1];

        std::vector<Point<double>> current_joint_positions;
        for (int joint = 0; joint < joint_count; ++joint) {
            current_joint_positions.push_back(Point<double>(
                joints(frame_idx, joint, 0), joints(frame_idx, joint, 1),
                joints(frame_idx, joint, 2)));
        }

        auto values = skeleton_filter.step(current_joint_positions, timestamps[frame_idx]);
        // std::cout << values[0] << std::endl;
        filtered_values.push_back(values);
    }
    return 0;

    // //////////////////////////////////////////////////////////
    // //////////////////////////////////////////////////////////
    // //////////////////////////////////////////////////////////
    // Only one axis!
    // //////////////////////////////////////////////////////////
    // //////////////////////////////////////////////////////////
    // //////////////////////////////////////////////////////////
    int axis = 0;
    for (int joint = 0; joint < joint_count; ++joint) {
        auto ad = Ad;
        auto b = A; // Won't be used anyway
        auto c = C;
        auto g = G;
        auto q = G; // Won't be used anyway
        auto m_measurement_noise = measurement_noise_for_all_joints(joint, axis); // focus on x axis

        GenericFilter1D<double> filter(ad, b, c, g, q, m_measurement_noise,
            system_noise_x, m_threshold);
        auto initial_state = MatrixXd(2, 1);
        initial_state(0, 0) = joints(0, joint, axis);
        initial_state(1, 0) = 0;
        auto initial_errors = MatrixXd(2, 2);
        initial_errors(0, 0) = 1;
        initial_errors(0, 1) = 0;
        initial_errors(1, 0) = 0;
        initial_errors(1, 1) = 1;
        filter.init(initial_state, initial_errors);

        int max_frame = n_frames;
        // We start with an offset of 1

        std::vector<double> filtered_values;
        // Add initial value
        filtered_values.push_back(joints(0, joint, axis));

        for (int frame_idx = 1; frame_idx < max_frame; ++frame_idx) {
            if (is_null[frame_idx])
                continue;
            double time_diff = timestamps[frame_idx] - timestamps[frame_idx - 1];
            auto value = filter.step(joints(frame_idx, joint, axis), time_diff);
            filtered_values.push_back(value);
        }

        std::ofstream file;
        file.open("out.csv");
        for (auto value : filtered_values) {
            file << value << "\r";
        }
        file << std::endl;
        break;
    }
    // for (auto timestamp : timestamps) {
    //     std::cout << timestamp << std::endl;
    // }
}

/*
void tensor()
{
    Tensor<double, 3> tensor(4, 10, 3);
    tensor(0, 0, 0) = 1;
    tensor(1, 0, 0) = 2;
    tensor(2, 0, 0) = 3;
    tensor(3, 0, 0) = 4;
    Eigen::array<Eigen::Index, 3> offsets = { 0, 0, 0 };
    Eigen::array<Eigen::Index, 3> extents = { 4, 1, 1 };
    std::cout << "A" << std::endl;
    std::cout << tensor.slice(offsets, extents).mean() << std::endl;
    std::cout << "B" << std::endl;
    offsets = { 0, 0, 0 };
    extents = { 1, 1, 3 };
    std::cout << tensor.slice(offsets, extents) << std::endl;
}
*/

/*
void mm()
{
    MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    std::cout << m << std::endl;
}
*/