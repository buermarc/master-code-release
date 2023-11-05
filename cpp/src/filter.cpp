#include "filter/adaptive/AdaptivePointFilter3D.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

#include <filter/ConstrainedSkeletonFilter.hpp>
#include <filter/SkeletonFilter.hpp>
#include <filter/Utils.hpp>
#include <filter/adaptive/AdaptiveBarShalomFilter1D.hpp>
#include <filter/adaptive/AdaptiveConstrainedSkeletonFilter.hpp>
#include <filter/adaptive/AdaptivePointFilter3D.hpp>
#include <filter/adaptive/AdaptiveRoseFilter1D.hpp>
#include <filter/adaptive/AdaptiveZarchanFilter1D.hpp>
#include <filter/com.hpp>

using json = nlohmann::json;

using Eigen::MatrixXd;
using Eigen::Tensor;

typedef AdaptivePointFilter3D<double, AdaptiveRoseFilter1D<double>> RosePointFilter;
typedef AdaptivePointFilter3D<double, AdaptiveBarShalomFilter1D<double>> BarPointFilter;
typedef AdaptivePointFilter3D<double, AdaptiveZarchanFilter1D<double>> ZarPointFilter;

std::string trimString(std::string str)
{
    const std::string whiteSpaces = " \t\n\r\f\v";
    // Remove leading whitespace
    size_t first_non_space = str.find_first_not_of(whiteSpaces);
    str.erase(0, first_non_space);
    // Remove trailing whitespace
    size_t last_non_space = str.find_last_not_of(whiteSpaces);
    str.erase(last_non_space + 1);
    return str;
}

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string> result;
    std::string line;
    std::getline(str, line);

    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
        result.push_back(trimString(cell));
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty()) {
        // If there was a trailing comma then add an empty element.
        result.push_back("");
    }
    return result;
}

void save_measurement_erros()
{
    // Instead of alway using a default file to get the measurement erros just
    // save them in hardcoded in a cpp file, so that they can be used as a
    // somewhat sane default, whithout wasting time recomputing them always.
    std::string var_path("../matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json");
    int joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);

    auto size = var.size();
    double* buffer = var.data();
    std::cout << "double buffer[] = {";
    for (int i = 0; i < size - 1; ++i) {
        std::cout << buffer[i] << ", ";
    }
    std::cout << buffer[size - 1] << "};" << std::endl;
}

void filter_reverse_pendelum()
{
    // TODO probably a 1x3 matrix
    Point<double> measurement_error(1., 0.5, 0.1);
    ZarPointFilter filter(measurement_error);
    std::vector<std::vector<Point<double>>> res;

    // Load csv
    std::ifstream csv_file("data/noisy_result.csv");

    auto header = getNextLineAndSplitIntoTokens(csv_file);

    auto results = getNextLineAndSplitIntoTokens(csv_file);
    std::cout << results.size() << std::endl;
    std::vector<double> values(results.size());
    std::transform(results.begin(), results.end(), values.begin(), [](auto element) { return std::stod(element); });

    std::vector<std::string> loop = { "com", "left", "right" };
    int i = 0;
    for (auto name : loop) {
        auto index = i * 3;
        std::vector<Point<double>> _res;

        Point<double> point(
            values[index],
            values[index + 1],
            values[index + 2]);
        filter.init(point);
        _res.push_back(point);
        res.push_back(_res);
        ++i;
    }

    while (!csv_file.eof()) {
        results = getNextLineAndSplitIntoTokens(csv_file);
        if (results[0] == "")
            break;
        std::transform(results.begin(), results.end(), values.begin(), [](auto element) {
            std::cout << element << std::endl;
            return std::stod(element);
        });
        i = 0;
        for (auto name : loop) {
            auto index = i * 3;

            Point<double> point(
                values[index],
                values[index + 1],
                values[index + 2]);
            double time = values.back();
            auto [filtered_point, _] = filter.step(point, time);
            res[i].push_back(filtered_point);
            ++i;
        }
    }

    // Write out CSV
    std::ofstream file;
    file.open("data/noisy_result_filtered.csv");

    // Write header
    std::for_each(loop.cbegin(), loop.cend() - 1, [&file](auto name) {
        file << name << "_x,";
        file << name << "_y,";
        file << name << "_z,";
    });
    auto name = loop.back();
    file << name << "_x,";
    file << name << "_y,";
    file << name << "_z";
    file << "\n";

    // Write elements
    int elements = res.front().size();
    for (int i = 0; i < elements; ++i) {
        bool first = true;
        for (int j = 0; j < loop.size(); ++j) {
            if (first) {
                file << res[j][i].x;
                first = false;
            } else {
                file << ", " << res[j][i].x;
            }
            file << ", " << res[j][i].y;
            file << ", " << res[j][i].z;
        }
        file << "\n";
    }
    file << std::endl;
}

int filter_data_with_constrained_skeleton_filter()
{
    std::string var_path("../matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json");
    int joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);

    std::string data_path("../matlab/sts_NFOV_UNBINNED_720P_30fps.json");
    auto [joints, n_frames, timestamps, is_null] = load_data(data_path, joint_count, 870);

    if (std::find(is_null.begin(), is_null.end(), true) != is_null.end()) {
        std::cerr << "found null" << std::endl;
    }

    std::vector<Point<double>> initial_points;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_points.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }
    // ConstrainedSkeletonFilter<double> filter(32, var);
    AdaptiveConstrainedSkeletonFilter<double, RosePointFilter> filter(32, var);
    filter.init(initial_points, timestamps[0]);

    std::vector<std::vector<Point<double>>> filtered_values;
    // Add initial value
    std::vector<Point<double>> initial_joints;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_joints.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }
    filtered_values.push_back(initial_joints);

    int max_frame = n_frames;
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

        auto start = std::chrono::high_resolution_clock::now();
        auto [values, _] = filter.step(current_joint_positions, timestamps[frame_idx]);
        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> time = stop - start;
        std::cerr << time.count() << "ms\n";
        // std::cout << values[0] << std::endl;
        std::cerr << current_joint_positions[0].x - values[0].x << std::endl;
        filtered_values.push_back(values);
    }

    // Write out filtere values into csv
    std::ofstream file;
    file.open("data/out.csv");

    // Write header
    int end = 32;
    for (int i = 0; i < end - 1; ++i) {
        file << "Joint_" << i << "_x,";
        file << "Joint_" << i << "_y,";
        file << "Joint_" << i << "_z,";
    }
    {
        int i = end - 1;
        file << "Joint_" << i << "_x";
        file << ",Joint_" << i << "_y";
        file << ",Joint_" << i << "_z";
    }
    file << "\n";

    // Write elements
    for (auto joints : filtered_values) {
        bool first = true;
        for (auto joint : joints) {
            if (first) {
                file << joint.x;
                first = false;
            } else {
                file << ", " << joint.x;
            }
            file << ", " << joint.y;
            file << ", " << joint.z;
        }
        file << "\n";
    }
    // file << "\r\n";
    file << std::endl;
    return 0;
}

int filter_data_with_skeleton_filter()
{
    std::cout << get_azure_kinect_com_matrix() << std::endl;
    return 0;
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

        auto [values, _] = skeleton_filter.step(current_joint_positions, timestamps[frame_idx]);
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
            auto [value, _] = filter.step(joints(frame_idx, joint, axis), time_diff);
            filtered_values.push_back(value);
        }

        std::ofstream file;
        file.open("data/other_out.csv");
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

int main()
{
    save_measurement_erros();
    // filter_reverse_pendelum();
    // filter_data_with_constrained_skeleton_filter();
}
