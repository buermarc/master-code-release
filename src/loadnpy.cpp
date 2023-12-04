#include <Eigen/Dense>
#include <cnpy/cnpy.h>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::MatrixXd;

int main(int argc, char** argv)
{
    auto npy_file = cnpy::npy_load(argv[1]);
    auto shape = npy_file.shape;
    auto data = npy_file.data<double>();
    auto matrix = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>>(data, shape.at(0), shape.at(1), shape.at(2));
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
}
