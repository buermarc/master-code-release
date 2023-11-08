#include <Eigen/Dense>
#include <filter/PointFilter3D.hpp>
#include <tuple>

using Eigen::MatrixXd;

std::tuple<MatrixXd, MatrixXd, MatrixXd> let_matricies_appear_magically()
{
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

    return { A, C, G };
}
