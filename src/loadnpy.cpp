#include "filter/ConstrainedSkeletonFilter.hpp"
#include "filter/GenericFilter1D.hpp"
#include <Eigen/Dense>
#include <cnpy/cnpy.h>
#include <filter/SkeletonFilter.hpp>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::MatrixXd;

int main(int argc, char** argv)
{
    double noise = 0.05;
    if (argc == 2) {
        noise = std::stod(argv[1]);
    }
    std::cout << "Noise: " << noise << std::endl;

    auto steps = cnpy::npy_load("data/steps.npy");
    auto noisy_steps = cnpy::npy_load("data/noisy_steps.npy");

    auto shape = steps.shape;
    auto data = steps.data<double>();
    auto noisy_data = noisy_steps.data<double>();

    auto matrix = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>>(data, shape.at(0), shape.at(1), shape.at(2));
    auto noisy_matrix = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>>(noisy_data, shape.at(0), shape.at(1), shape.at(2));
    std::cout << matrix << std::endl;
    std::cout << "shape" << std::endl;
    std::cout << shape.at(0) << std::endl;
    std::cout << shape.at(1) << std::endl;
    std::cout << shape.at(2) << std::endl;
    std::cout << matrix(0, 0, 0) << std::endl;
    std::cout << matrix(0, 0, 1) << std::endl;
    std::cout << matrix(0, 0, 2) << std::endl;
    std::cout << matrix(0, 1, 0) << std::endl;
    std::cout << matrix(0, 1, 1) << std::endl;
    std::cout << matrix(0, 1, 2) << std::endl;
    std::cout << matrix(0, 2, 0) << std::endl;
    std::cout << matrix(0, 2, 1) << std::endl;
    std::cout << matrix(0, 2, 2) << std::endl;

    MatrixXd ones = MatrixXd::Ones(shape.at(1), shape.at(2)) * noise;
    MatrixXd zero = MatrixXd::Zero(1, shape.at(1));
    SkeletonFilter<double> filter = SkeletonFilter<double>(11, ones, zero);

    Eigen::Tensor filtered = Eigen::Tensor<double, 3, Eigen::RowMajor>(shape.at(0), shape.at(1), shape.at(2));

    std::vector<std::vector<Point<double>>> positions;
    std::vector<std::vector<Point<double>>> velocities;
    std::vector<Point<double>> points;

    for (int i = 0; i < shape.at(1); ++i) {
        points.push_back(Point<double>(
            noisy_matrix(0, i, 0),
            noisy_matrix(0, i, 1),
            noisy_matrix(0, i, 2)
        ));
        filtered(0, i, 0) = noisy_matrix(0, i, 0);
        filtered(0, i, 1) = noisy_matrix(0, i, 1);
        filtered(0, i, 2) = noisy_matrix(0, i, 2);
    }
    double time = 0;
    filter.init(points, time);

    for (int i = 1; i < shape.at(0); ++i) {
        time += 33e-3;
        points.clear();
        for (int j = 0; j < shape.at(1); ++j) {
            points.push_back(Point<double>(
                noisy_matrix(i, j, 0),
                noisy_matrix(i, j, 1),
                noisy_matrix(i, j, 2)
            ));
        }
        auto [positions, velocity] = filter.step(points, time);
        for (int j = 0; j < shape.at(1); ++j) {
            auto position = positions.at(j);
            filtered(i, j, 0) = position.x;
            filtered(i, j, 1) = position.y;
            filtered(i, j, 2) = position.z;
        }
    }

    auto diff_true_filtered = matrix - filtered;
    auto diff_noise_filtered = noisy_matrix - filtered;
    std::cout << "diff_true_filtered" << std::endl;
    std::cout << diff_true_filtered << std::endl;
    std::cout << "diff_noise_filtered" << std::endl;
    std::cout << diff_noise_filtered << std::endl;
    auto d = filtered.dimensions();
    cnpy::npy_save("data/filtered.npy", filtered.data(), {d[0], d[1], d[2]}, "w");
    std::cout << ones << std::endl;
}
