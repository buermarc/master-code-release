#include <Eigen/Dense>
#include <filter/ConstrainedSkeletonFilter.hpp>

using Eigen::MatrixXd;

MatrixXd generate_rigid_joint_al()
{
    MatrixXd Al(18, 18);
    MatrixXd first(9, 18);
    MatrixXd second(9, 18);
    first << MatrixXd::Zero(9, 9), MatrixXd::Identity(9, 9);
    second << MatrixXd::Zero(9, 9), MatrixXd::Zero(9, 9);

    Al << first, second;
    return Al;
}
