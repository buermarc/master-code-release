#include <Eigen/Dense>
#include <filter/ConstrainedSkeletonFilter.hpp>
#include <filter/PointFilter3D.hpp>
#include <filter/com.hpp>
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions)
{
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}

TEST(PointConstructor, BasicAssertions)
{
    double test[3] = {1,2,3};
    Point<double> value(test);
    EXPECT_EQ(value.x, 1);
    EXPECT_EQ(value.y, 2);
    EXPECT_EQ(value.z, 3);
}


TEST(CenterOfMassTest, BasicAssertions)
{
    // Expect two strings not to be equal.
    MatrixXd MM = get_azure_kinect_com_matrix();
    EXPECT_EQ(MM.size(), (1, 32));
}

TEST(EigenLearningTests, BasicAssertions)
{
    // Expect two strings not to be equal.
    MatrixXd temp(9, 18);
    MatrixXd eye = MatrixXd::Identity(9, 9);
    MatrixXd zeros = MatrixXd::Zero(9, 9);

    temp << eye, zeros;
    EXPECT_EQ(temp.rows(), 9);
    EXPECT_EQ(temp.cols(), 18);
    EXPECT_EQ(temp(0, 0), 1);
    EXPECT_EQ(temp(8, 8), 1);
    EXPECT_EQ(temp(8, 9), 0);
}

TEST(RigidJointConstructFilter3InitTests, BasicAssertions)
{

    std::vector<int> example_joints = { 19, 20, 21 }; // left leg
    MatrixXd Al(18, 18);
    MatrixXd Cl(9, 18);
    MatrixXd Gl(18, 9);
    // Gld = [(Ts^2/2)*eye(9,9); Ts*eye(9,9)]; % considering that acceleration has normal ditribution with zero mean, See truck example in wikipedia and Machthaler und Dingler (2017)
    Al = generate_rigid_joint_al();
    EXPECT_EQ(Al.rows(), 18);
    EXPECT_EQ(Al.cols(), 18);

    Cl << MatrixXd::Identity(9, 9), MatrixXd::Zero(9, 9);
    EXPECT_EQ(Cl.rows(), 9);
    EXPECT_EQ(Cl.cols(), 18);
    EXPECT_EQ(Cl(0, 0), 1);
    EXPECT_EQ(Cl(0, 9), 0);

    Gl << MatrixXd::Identity(9, 9), MatrixXd::Identity(9, 9);
    EXPECT_EQ(Gl.rows(), 18);
    EXPECT_EQ(Gl.cols(), 9);
    EXPECT_EQ(Gl(0, 0), 1);
    EXPECT_EQ(Gl(8, 8), 1);
    EXPECT_EQ(Gl(9, 0), 1);
    EXPECT_EQ(Gl(17, 8), 1);

    MatrixXd phi_1(9, 9);
    MatrixXd phi_2(9, 9);

    auto eye = [](int size) { return MatrixXd::Identity(size, size); };
    auto zero = [](int size) { return MatrixXd::Zero(size, size); };

    phi_1 << eye(3), -1 * eye(3), zero(3), -1 * eye(3), eye(3), zero(3), zero(3), zero(3), zero(3);
    phi_2 << zero(3), zero(3), zero(3), zero(3), eye(3), -1 * eye(3), zero(3), -1 * eye(3), eye(3);

    EXPECT_EQ(phi_1.rows(), 9);
    EXPECT_EQ(phi_1.cols(), 9);
    EXPECT_EQ(phi_1(0, 0), 1);
    EXPECT_EQ(phi_1(3, 0), -1);
    EXPECT_EQ(phi_1(0, 3), -1);
    EXPECT_EQ(phi_1(3, 3), 1);
    EXPECT_EQ(phi_1(8, 8), 0);

    EXPECT_EQ(phi_1.reverse(), phi_2);

    std::string var_path(std::format("/home/{}/repos/master/code/_matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json", std::getenv("USER")));
    auto joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);

    MatrixXd measurement_noise = zero(3);
    {
        int i = 0;
        for (auto element : example_joints) {
            measurement_noise.row(i) = 10 * var.row(element).array().sqrt();
            ++i;
        }
    } // remove i from leaking into outer scope

    double factor_system_noise = 1.0 / 3;
    double vmax = 10.0 * factor_system_noise;
    double sigma_system_noise = vmax / 3;
    MatrixXd system_noise = eye(9) * std::pow(sigma_system_noise, 2);

    EXPECT_EQ(system_noise.rows(), 9);
    EXPECT_EQ(system_noise.cols(), 9);
    EXPECT_EQ(system_noise(0, 1), 0);
    EXPECT_EQ(system_noise(0, 0), std::pow(sigma_system_noise, 2));
    EXPECT_EQ(system_noise(8, 8), std::pow(sigma_system_noise, 2));

    double threshold = 2;

    auto sub_ad = [](MatrixXd Ad, double time_diff) {
        MatrixXd result(18, 18);
        MatrixXd first = Ad.array() * time_diff;
        MatrixXd second = ((1 - Ad.array()) * Ad.exp().array());
        result = first + second;
        return result;
    };

    MatrixXd aldn;
    aldn = sub_ad(Al, 0.2);

    auto sub_gd = [](MatrixXd Gd, double time_diff) {
        MatrixXd result(18, 9);
        MatrixXd first = MatrixXd::Identity(9, 9) * (std::pow(time_diff, 2) / 2);
        MatrixXd second = MatrixXd::Identity(9, 9) * time_diff;
        result << first, second;
        return result;
    };

    MatrixXd gldn;
    gldn = sub_gd(Gl, 0.2); // Gd is unnessary, but IDGAF

    auto filter = RigidJointConstructFilter3<double>(
        Al,
        Cl,
        Gl,
        phi_1,
        phi_2,
        measurement_noise,
        system_noise,
        threshold,
        sub_ad,
        sub_gd,
        example_joints);

    auto initial_state = MatrixXd::Constant(18, 1, 1);
    EXPECT_EQ(initial_state.rows(), 18);
    EXPECT_EQ(initial_state.cols(), 1);
    EXPECT_EQ(initial_state(0, 0), 1);
    EXPECT_EQ(initial_state(8, 0), 1);
    EXPECT_EQ(initial_state(17, 0), 1);

    auto initial_errors = MatrixXd::Identity(18, 18);

    filter.init(initial_state, initial_errors);

    // % Measurement vector: y = [px1; py1; pz1; px2; py2; pz2; px3; py3; pz3]
    MatrixXd measurement(9, 1);
    measurement << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

    auto result = filter.step(measurement, 0.5);
    for (int k = 0; k < 0 * 100; ++k) {
        result = filter.step(measurement, 0.5);
    }
}

TEST(RigidJointConstructFilter3InitTestsDefaultInit, BasicAssertions)
{

    std::string var_path(std::format("/home/{}/repos/master/code/_matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json", std::getenv("USER")));
    auto joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);

    std::vector<int> joints = { 19, 20, 21 };
    auto filter = RigidJointConstructFilter3<double>::default_init(joints, var);

    auto initial_state = MatrixXd::Constant(18, 1, 1);
    auto initial_errors = MatrixXd::Identity(18, 18);

    filter.init(initial_state, initial_errors);

    // % Measurement vector: y = [px1; py1; pz1; px2; py2; pz2; px3; py3; pz3]
    MatrixXd measurement(9, 1);
    measurement << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

    auto result = filter.step(measurement, 0.5);
    for (int k = 0; k < 0 * 100; ++k) {
        result = filter.step(measurement, 0.5);
    }
}

TEST(ConstrainedSkeletonFilterInit, BasicAssertions)
{

    std::string var_path(std::format("/home/{}/repos/master/code/_matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json", std::getenv("USER")));
    auto joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);

    auto filter = ConstrainedSkeletonFilter<double>(
        joint_count,
        var,
        get_azure_kinect_com_matrix());

    std::vector<Point<double>> initial_points;
    for (int i = 0; i < 32; ++i) {
        initial_points.push_back(Point(1.0 + i, 1.0 + i, 1.0 + i));
    }

    double initial_time = 0.0;
    filter.init(initial_points, initial_time);

    double next_time = 1.0;
    std::vector<Point<double>> joints;
    for (int i = 0; i < 32; ++i) {
        joints.push_back(Point(5.0 + i, 5.0 + i, 5.0 + i));
    }
    filter.step(joints, next_time);
}

TEST(PointFilter3DTest, BasicAssertions)
{

    std::string var_path(std::format("/home/{}/repos/master/code/_matlab/stand_b2_t1_NFOV_UNBINNED_720P_30fps.json", std::getenv("USER")));
    auto joint_count = 32;
    auto [var_joints, _n_frames, _timestamps, _is_null] = load_data(var_path, joint_count);
    auto var = get_measurement_error(var_joints, joint_count, 209, 339);

    std::string data_path(std::format("/home/{}/repos/master/code/_matlab/sts_NFOV_UNBINNED_720P_30fps.json", std::getenv("USER")));
    auto [joints, n_frames, timestamps, is_null] = load_data(data_path, joint_count, 870);
    auto filter = PointFilter3D<double>::default_init(17, var);

    std::vector<Point<double>> initial_points;
    for (int joint = 0; joint < joint_count; ++joint) {
        initial_points.push_back(Point<double>(
            joints(0, joint, 0), joints(0, joint, 1), joints(0, joint, 2)));
    }

    auto joint_idx = 17;
    auto initial_point = initial_points[joint_idx];
    filter.init(initial_point);

    std::ofstream file;
    file.open("data/point3d.csv");
    file << "Joint_" << joint_idx << "_x,";
    file << "Joint_" << joint_idx << "_y,";
    file << "Joint_" << joint_idx << "_z";
    file << "\n";
    file << initial_point.x << ",";
    file << initial_point.y << ",";
    file << initial_point.z << "\n";

    for (int i = 1; i < n_frames; ++i) {
        auto point = Point<double>(
            joints(i, joint_idx, 0),
            joints(i, joint_idx, 1),
            joints(i, joint_idx, 2));
        auto time_diff = timestamps[i] - timestamps[i - 1];
        auto [position, velocity] = filter.step(point, time_diff);
        file << position.x << ",";
        file << position.y << ",";
        file << position.z << "\n";
    }
}
