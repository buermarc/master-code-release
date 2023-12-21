#include "filter/ConstrainedSkeletonFilter.hpp"
#include "filter/GenericFilter1D.hpp"
#include <Eigen/Dense>
#include <cnpy/cnpy.h>
#include <filter/SkeletonFilter.hpp>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::MatrixXd;

int main(int argc, char** argv)
{
    auto steps = cnpy::npy_load("simulations/E3/steps.npy");
    auto noisy_steps = cnpy::npy_load("simulations/E3/steps.npy");

    auto shape = steps.shape;
    auto data = steps.data<double>();
    auto noisy_data = noisy_steps.data<double>();
    auto matrix = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>>(data, shape.at(0), shape.at(1), shape.at(2));
    auto noisy_matrix = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>>(data, shape.at(0), shape.at(1), shape.at(2));
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

    MatrixXd ones = MatrixXd::Ones(shape.at(1), shape.at(2));
    MatrixXd zero = MatrixXd::Zero(1, shape.at(1));
    SkeletonFilter<double> filter = SkeletonFilter<double>(11, ones, zero);

    std::vector<std::vector<Point<double>>> positions;
    std::vector<std::vector<Point<double>>> velocities;
    std::vector<Point<double>> points;

    for (int i = 0; i < shape.at(1); ++i) {
        points.push_back(Point<double>(
            matrix(0, i, 0),
            matrix(0, i, 1),
            matrix(0, i, 2)
        ));
    }
    double time = 0;
    filter.init(points, time);

    for (int i = 1; i < shape.at(0); ++i) {
        time += 33e-3;
        points.clear();
        for (int j = 0; j < shape.at(j); ++i) {
            points.push_back(Point<double>(
                matrix(i, j, 0),
                matrix(i, j, 1),
                matrix(i, i, 2)
            ));
        }
        auto [position, velocity] = filter.step(points, time);
    }
}
