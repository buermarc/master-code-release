#pragma once
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>

using Eigen::MatrixXd;

enum AZURE_JOINT {
    PELVIS,
    SPINE_NAVEL,
    SPINE_CHEST,
    NECK,
    CLAVICLE_LEFT,
    SHOULDER_LEFT,
    ELBOW_LEFT,
    WRIST_LEFT,
    HAND_LEFT,
    HANDTIP_LEFT,
    THUMB_LEFT,
    CLAVICLE_RIGHT,
    SHOULDER_RIGHT,
    ELBOW_RIGHT,
    WRIST_RIGHT,
    HAND_RIGHT,
    HANDTIP_RIGHT,
    THUMB_RIGHT,
    HIP_LEFT,
    KNEE_LEFT,
    ANKLE_LEFT,
    FOOT_LEFT,
    HIP_RIGHT,
    KNEE_RIGHT,
    ANKLE_RIGHT,
    FOOT_RIGHT,
    HEAD,
    NOSE,
    EYE_LEFT,
    EAR_LEFT,
    EYE_RIGHT,
    EAR_RIGHT,
};

void _one_origin_two_joints(
    MatrixXd& MM,
    double normalized_mass,
    double normalized_com_position,
    int origin_joint,
    int joint_1,
    int joint_2)
{
    MM(0, joint_1) += normalized_mass * normalized_com_position / 2;
    MM(0, joint_2) += normalized_mass * normalized_com_position / 2;
    MM(0, origin_joint) += normalized_mass * (1 - normalized_com_position);
}

void _two_origins_one_joint(
    MatrixXd& MM,
    double normalized_mass,
    double normalized_com_position,
    int origin_joint_1,
    int origin_joint_2,
    int joint)
{
    MM(0, joint) += normalized_mass * normalized_com_position;
    MM(0, origin_joint_1) += normalized_mass * (1 - normalized_com_position) / 2;
    MM(0, origin_joint_2) += normalized_mass * (1 - normalized_com_position) / 2;
}

void _one_origin_one_joint(
    MatrixXd& MM,
    double normalized_mass,
    double normalized_com_position,
    int origin_joint,
    int joint)
{
    MM(0, joint) += normalized_mass * normalized_com_position;
    MM(0, origin_joint) += normalized_mass * (1 - normalized_com_position);
}

MatrixXd get_azure_kinect_com_matrix()
{
    MatrixXd MM(1, 32);

    for (int i = 0; i < 32; ++i) {
        MM(0, i) = 0; // Initialize with zero
    }

    _one_origin_two_joints(MM, 0.1117, 0.6115, SPINE_NAVEL, HIP_LEFT, HIP_RIGHT); // 1
    _one_origin_one_joint(MM, 0.1633, 0.4502, SPINE_CHEST, SPINE_NAVEL); // 2
    _two_origins_one_joint(MM, 0.1596, 0.2999, CLAVICLE_LEFT, CLAVICLE_RIGHT, SPINE_CHEST); // 3
    _one_origin_two_joints(MM, 0.0694, 2. / 3., HEAD, EAR_LEFT, EAR_RIGHT); // 4
    _one_origin_one_joint(MM, 0.0271, 0.5772, SHOULDER_LEFT, ELBOW_LEFT); // 5
    _one_origin_one_joint(MM, 0.0162, 0.4574, ELBOW_LEFT, WRIST_LEFT); // 6
    _one_origin_one_joint(MM, 0.0061, 0.79, WRIST_LEFT, HAND_LEFT); // 7
    _one_origin_one_joint(MM, 0.0271, 0.5772, SHOULDER_RIGHT, ELBOW_RIGHT); // 8
    _one_origin_one_joint(MM, 0.0162, 0.4574, ELBOW_RIGHT, WRIST_RIGHT); // 9
    _one_origin_one_joint(MM, 0.0061, 0.79, WRIST_RIGHT, HAND_RIGHT); // 10
    _one_origin_one_joint(MM, 0.1416, 0.4095, HIP_LEFT, KNEE_LEFT); // 11
    _one_origin_one_joint(MM, 0.0433, 0.4395, KNEE_LEFT, ANKLE_LEFT); // 12
    _one_origin_one_joint(MM, 0.0137, (4. / 3.) * (0.4415 - 1. / 4.), ANKLE_LEFT, FOOT_LEFT); // 13
    _one_origin_one_joint(MM, 0.1416, 0.4095, HIP_RIGHT, KNEE_RIGHT); // 14
    _one_origin_one_joint(MM, 0.0433, 0.4395, KNEE_RIGHT, ANKLE_RIGHT); // 15
    _one_origin_one_joint(MM, 0.0137, (4. / 3.) * (0.4415 - 1. / 4.), ANKLE_RIGHT, FOOT_RIGHT); // 16

    return MM;
}
