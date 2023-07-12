#include <cstddef>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <array>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

using Eigen::MatrixXd;
using Eigen::Tensor;

void mm() {
    MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    std::cout << m << std::endl;
}

std::tuple<MatrixXd, MatrixXd, MatrixXd>  let_matricies_appear_magically() {
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

    return {A, C, G};
}

std::tuple<Tensor<double, 3>, int, std::vector<double>, std::vector<bool>> load_data(std::string path, int joint_counts, int max_frames = -1) {
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

    return {joints, n_frames, timestamps, is_null};
}
MatrixXd get_measurement_error(Tensor<double, 3> joints, int joint_counts, int frame_start, int frame_end) {

    MatrixXd var(joint_counts, 3);
    Eigen::array<Eigen::Index, 3> offsets;
    Eigen::array<Eigen::Index, 3> extents;
    Tensor<double, 0> mean_t;
    Tensor<double, 0> sum_t;
    for (int i = 0; i < joint_counts; ++i) {
        for (int j = 0; j < 2; ++j) {
            offsets = {frame_start, i, j};
            extents = {(frame_end-frame_start)+1, 1, 1};
            mean_t = joints.slice(offsets, extents).mean();
            sum_t = (joints.slice(offsets, extents) - mean_t(0)).pow(2).sum();
            var(i, j) = sum_t(0) / (frame_end-frame_start);
        }
        int j = 2;
        offsets = {frame_start, i, j};
        extents = {(frame_end-frame_start)+1, 1, 1};
        mean_t = joints.slice(offsets, extents).mean();
        sum_t = (joints.slice(offsets, extents) - mean_t(0)).pow(2).sum();
        var(i, j) = (sum_t(0) / (frame_end-frame_start) / 10);  // treat 'z' as less important
    }
    return var;
}


void tensor() {
    Tensor<double, 3> tensor(4, 10, 3);
    tensor(0, 0, 0) = 1;
    tensor(1, 0, 0) = 2;
    tensor(2, 0, 0) = 3;
    tensor(3, 0, 0) = 4;
    Eigen::array<Eigen::Index, 3> offsets = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> extents = {4, 1, 1};
    std::cout << "A" << std::endl;
    std::cout << tensor.slice(offsets, extents).mean() << std::endl;
    std::cout << "B" << std::endl;
    offsets = {0, 0, 0};
    extents = {1, 1, 3};
    std::cout << tensor.slice(offsets, extents) << std::endl;
}

void filter() {
    /**
     * what to do:
     * where were the formulas:
     */
    double time_diff = 0.4;
}

/**
 * how should the class look like:
 * could do it specific for 3d
 * could also do it just for any value
 *
 * If we are not generic then we can set Adn and the rest of the matrices to a fixed value
 * if we want to be generic this would be a bit trickier
 *
 * Maybe Generic class wich takes the stuff and actual clas fo rour case where we set the stuff in the constructor
 * could also do it just for any value
 *
 * Basically the gated 3d case just sets 3 GenericFilters for each var and then
 * steps with the 3d thing
 *
 *
 * Intgeration things might be the stuff that a specific class needs to
 * implement, because I do not want to have to implement symbolic integration 
 */
template<typename Value>
class GenericFilter1D {
    Value value;
    MatrixXd corrected_state; // 2x1
    MatrixXd corrected_errors; // 2x1
    MatrixXd Ad;
    MatrixXd B;
    MatrixXd G;
    MatrixXd Q;  // Q(k) = Var(z(k))

    MatrixXd C;
    MatrixXd CT;

    Value measurement_noise;
    Value system_noise;
    Value threshold;

    public:
    GenericFilter1D () {};

    GenericFilter1D (
        MatrixXd ad,
        MatrixXd b,
        MatrixXd c,
        MatrixXd g,
        MatrixXd q,
        Value m_measurement_noise,
        Value m_system_noise,
        Value m_threshold
    ) {
        Ad = ad;
        B = b;
        C = c;
        G = g;
        Q = q;
        measurement_noise = std::pow(m_measurement_noise, 2);
        system_noise = m_system_noise;
        threshold = m_threshold;

        CT = C.transpose();
    };

    void init(MatrixXd initial_state, MatrixXd initial_errors) {
        // how do they look and where do they come from
        // look:
        // think was 2x1
        // origin:
        // just the initial value that is recorded
        corrected_state = initial_state;
        corrected_errors = initial_errors;
    }

    Value step(Value value, Value time_diff) {

        // Hard coded sub A = [1, Ts; 0, 1]
        MatrixXd Adn = Ad.replicate(1, 1);
        Adn(0, 1) = time_diff;

        // Again: hard coded sub G = [Ts^2/2; Ts]
        MatrixXd Gdn = G.replicate(1, 1);
        Gdn(0, 0) = std::pow(time_diff, 2) / 2;
        Gdn(1, 0) = time_diff;

        auto AdnT = Adn.transpose();
        auto GdnT = Gdn.transpose();

        MatrixXd predicted_state = Adn * corrected_state;
        MatrixXd predicted_errors = Adn * corrected_errors * AdnT + Gdn * system_noise * GdnT;

        auto comb = C * predicted_state;
        Value innovation = value - (C * predicted_state).array()(0);
        Value innovation_covariance = (C * Adn * predicted_errors * AdnT * CT).array()(0) + (C * system_noise * CT).array()(0) + measurement_noise;
        Value sigma_value = std::sqrt(innovation_covariance);
        Value innovation_norm = innovation / sigma_value;

        MatrixXd measurement_noise_matrix(1, 1);
        measurement_noise_matrix(0,  0) = measurement_noise;
        if (innovation_norm < threshold) {
            MatrixXd tmp = C * predicted_errors * CT + measurement_noise_matrix;
            auto pseudo_inv = tmp.completeOrthogonalDecomposition().pseudoInverse();
            MatrixXd K_value = predicted_errors * CT * pseudo_inv;
            corrected_state = predicted_state + K_value * (value - (C * predicted_state).array()(0));
            auto eye = MatrixXd::Identity(2, 2);
            corrected_errors = (eye - K_value * C) * predicted_errors;
        } else {
            corrected_state = predicted_state;
            corrected_errors = predicted_errors;
        }
        return corrected_state(0, 0);
    }
};

template<typename Value> 
class Point {
    public:
    Value x;
    Value y;
    Value z;

    Point() {
        x=0;
        y=0;
        z=0;
    }
    Point(Value m_x, Value m_y, Value m_z) : x(m_x), y(m_y), z(m_z) {}
    
    template<typename U> 
    friend std::ostream& operator<<(std::ostream& out, const Point<U>& point);
};

template<typename Value> 
std::ostream& operator<<(std::ostream& out, const Point<Value>& point)
{
    // Since operator<< is a friend of the Point class, we can access Point's members directly.
    out << "Point(" << point.x << ", " << point.y << ", " << point.z << ')'; // actual output done here

    return out;
}

template<typename Value> 
class PointFilter3D {
    GenericFilter1D<Value> x_filter;
    GenericFilter1D<Value> y_filter;
    GenericFilter1D<Value> z_filter;

    
    public:

    PointFilter3D(Point<Value> measurement_noise, Point<Value> system_noise, Value threshold) {
        auto [ad, c, g] = let_matricies_appear_magically();
        x_filter = GenericFilter1D<Value>(
            ad,
            ad,  // FIXME: we do not care
            c,
            g,
            g,  // FIXME: we do not care
            measurement_noise.x,
            system_noise.x,
            threshold
        );
        y_filter  = GenericFilter1D<Value>(
            ad,
            ad,  // FIXME: we do not care
            c,
            g,
            g,  // FIXME: we do not care
            measurement_noise.y,
            system_noise.y,
            threshold
        );
        z_filter  = GenericFilter1D<Value>(
            ad,
            ad,  // FIXME: we do not care
            c,
            g,
            g,  // FIXME: we do not care
            measurement_noise.z,
            system_noise.z,
            threshold
        );
    }

    void init(Point<Value> initial_point) {
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

    Point<Value> step(Point<Value> value, Value time_diff) {
        Point<Value> result;
        result.x = x_filter.step(value.x, time_diff);
        result.y = y_filter.step(value.y, time_diff);
        result.z = z_filter.step(value.z, time_diff);
        return result;
    }
    // Consider a point an outlier if any of the axis are outliers
    // How to include confidence level
};


template<typename Value> 
class SkeletonFilter {
    size_t n_joints;
    bool initialized = false;
    Value last_time;

    std::vector<PointFilter3D<Value>> joint_filters;

    public:

    int joint_count() {
        return n_joints;
    }

    SkeletonFilter(
        std::vector<Point<Value>> measurement_noises,
        std::vector<Point<Value>> system_noises,
        int m_n_joints,
        Value threshold
    ) : n_joints(m_n_joints) {
        for (int i = 0; i < m_n_joints; ++i) {
            auto filter = PointFilter3D<Value>(measurement_noises[i], system_noises[i], threshold);
            joint_filters.push_back(filter);
        }
    }

    void init(std::vector<Point<Value>> inital_points, Value initial_time) {
        if (initialized) {
            return;
        }
        for (int i = 0; i < n_joints; ++i) {
            joint_filters[i].init(inital_points[i]);
        }
        last_time = initial_time;
        initialized = true;
    }

    bool is_initialized() {
        return initialized;
    }

    std::vector<Point<Value>> step(std::vector<Point<Value>> values, Value new_time) {
        std::vector<Point<Value>> results;
        auto time_diff = new_time - last_time;
        // FIXME: Not nice using a 0..n_joints loop and push_back at the same time
        for (int i = 0; i < n_joints; ++i) {
            results.push_back(joint_filters[i].step(values[i], time_diff));
        }
        return results;
    }
};

template<typename Value> 
class SkeletonFilterBuilder {
    
    std::string noise_data_path;
    int joint_count;
    std::vector<Point<Value>> measurement_noises;
    std::vector<Point<Value>> system_noises;
    Value threshold;

    public:
    SkeletonFilterBuilder(
        std::string m_noise_data_path,
        int m_joint_count,
        Value m_threshold
    ) : joint_count(m_joint_count) {
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
            m_measurement_noises.push_back(Point<Value>(
                measurement_noise_for_all_joints(joint, 0),
                measurement_noise_for_all_joints(joint, 1),
                measurement_noise_for_all_joints(joint, 2)
            ));
            m_system_noises.push_back(Point<Value>(
                system_noise_x,
                system_noise_x,
                system_noise_x
            ));
        }

        measurement_noises = m_measurement_noises;
        system_noises = m_system_noises;
    }

    SkeletonFilter<Value> build() {
        return SkeletonFilter<Value>(measurement_noises, system_noises, joint_count, threshold);
    }
};



int main()
{
    std::string var_path("../matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json");
    int joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);

    std::string data_path("../matlab/sts_NFOV_UNBINNED_720P_30fps.json");
    auto [joints, n_frames, timestamps, is_null] = load_data(data_path, joint_count, 870);

    if (std::find(is_null.begin(), is_null.end(), true) != is_null.end() ) {
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
    // Gd = simplify(int(expm(Ts*A)*G)); % fuer reale Abtastung (Abtast-Halte-Glied)
    // auto Gd = Ad * G;
    // We have to hard code the integration
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
        measurement_noises.push_back(Point<double>(
            measurement_noise_for_all_joints(joint, 0),
            measurement_noise_for_all_joints(joint, 1),
            measurement_noise_for_all_joints(joint, 2)
        ));
        system_noises.push_back(Point<double>(
            system_noise_x,
            system_noise_x,
            system_noise_x
        ));
        initial_points.push_back(Point<double>(
            joints(0, joint, 0),
            joints(0, joint, 1),
            joints(0, joint, 2)
        ));
    }
    SkeletonFilter<double> skeleton_filter(measurement_noises, system_noises, 32, m_threshold);
    skeleton_filter.init(initial_points, timestamps[0]);

    std::vector<std::vector<Point<double>>> filtered_values;
    // Add initial value
    std::vector<Point<double>> initial_joints;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_joints.push_back(Point<double>(
            joints(0, joint, 0),
            joints(0, joint, 1),
            joints(0, joint, 2)
        ));
    }
    filtered_values.push_back(initial_joints);

    int max_frame = n_frames;
    std::cout << "n_frames " << n_frames << std::endl;
    for (int frame_idx = 1; frame_idx < max_frame; ++frame_idx) {
        if (is_null[frame_idx])
            continue;
        //double time_diff = timestamps[frame_idx] - timestamps[frame_idx-1];

        std::vector<Point<double>> current_joint_positions;
        for (int joint = 0; joint < joint_count; ++joint) {
            current_joint_positions.push_back(Point<double>(
                joints(frame_idx, joint, 0),
                joints(frame_idx, joint, 1),
                joints(frame_idx, joint, 2)
            ));
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
        auto m_measurement_noise = measurement_noise_for_all_joints(joint, axis);  // focus on x axis

        GenericFilter1D<double> filter(
            ad,
            b,
            c,
            g,
            q,
            m_measurement_noise,
            system_noise_x,
            m_threshold
        );
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
            double time_diff = timestamps[frame_idx] - timestamps[frame_idx-1];
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
