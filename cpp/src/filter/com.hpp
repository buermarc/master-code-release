#pragma once
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <tuple>

#include "filter/Point.hpp"
using Eigen::MatrixXd;

template <typename Value>
class Plane {
    Point<Value> a, b, c, d;

    Plane(Point<Value> m_a,
        Point<Value> m_b,
        Point<Value> m_c,
        Point<Value> m_d)
        : a(m_a)
        , b(m_b)
        , c(m_c)
        , d(m_d) {};

public:
    std::tuple<Point<Value>, Point<Value>> into_normal_and_center_point()
    {
        Eigen::Vector3f va(a.x, a.y, a.z);
        Eigen::Vector3f vb(b.x, b.y, b.z);
        Eigen::Vector3f vc(c.x, c.y, c.z);
        Eigen::Vector3f vd(d.x, d.y, d.z);

        auto vml = va + 0.5 * (vd - va);
        auto vmr = vb + 0.5 * (vc - vb);

        auto vcenter = vml + 0.5 * (vmr - vml);

        auto vnorm = (vcenter - va).cross(vcenter - vd);

        Point<Value> center(vcenter(0), vcenter(1), vcenter(2));
        Point<Value> norm(vnorm(0), vnorm(1), vnorm(2));
        return std::make_tuple(center, norm);
    }
};

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

enum SEX {
    MALE,
    FEMALE,
    AVERAGE,
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

template <typename Value>
Plane<Value> azure_kinect_bos(std::vector<Point<Value>> joints)
{
    return Plane(joints[ANKLE_LEFT], joints[FOOT_LEFT], joints[ANKLE_RIGHT], joints[FOOT_RIGHT]);
}

MatrixXd get_azure_kinect_com_matrix(SEX sex = AVERAGE)
{
    MatrixXd MM(1, 32);

    for (int i = 0; i < 32; ++i) {
        MM(0, i) = 0; // Initialize with zero
    }

    switch (sex) {
    case MALE:
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
        break;
    case FEMALE:
        _one_origin_two_joints(MM, 0.1247, 0.492, SPINE_NAVEL, HIP_LEFT, HIP_RIGHT); // 1
        _one_origin_one_joint(MM, 0.1465, 0.4512, SPINE_CHEST, SPINE_NAVEL); // 2
        _two_origins_one_joint(MM, 0.1545, 0.2077, CLAVICLE_LEFT, CLAVICLE_RIGHT, SPINE_CHEST); // 3
        _one_origin_two_joints(MM, 0.0668, 2. / 3., HEAD, EAR_LEFT, EAR_RIGHT); // 4
        _one_origin_one_joint(MM, 0.0255, 0.5754, SHOULDER_LEFT, ELBOW_LEFT); // 5
        _one_origin_one_joint(MM, 0.0138, 0.4559, ELBOW_LEFT, WRIST_LEFT); // 6
        _one_origin_one_joint(MM, 0.0056, 0.7474, WRIST_LEFT, HAND_LEFT); // 7
        _one_origin_one_joint(MM, 0.0255, 0.5754, SHOULDER_RIGHT, ELBOW_RIGHT); // 8
        _one_origin_one_joint(MM, 0.0138, 0.4559, ELBOW_RIGHT, WRIST_RIGHT); // 9
        _one_origin_one_joint(MM, 0.0056, 0.7474, WRIST_RIGHT, HAND_RIGHT); // 10
        _one_origin_one_joint(MM, 0.1478, 0.3612, HIP_LEFT, KNEE_LEFT); // 11
        _one_origin_one_joint(MM, 0.0481, 0.4352, KNEE_LEFT, ANKLE_LEFT); // 12
        _one_origin_one_joint(MM, 0.0129, (4. / 3.) * (0.4014 - 1. / 4.), ANKLE_LEFT, FOOT_LEFT); // 13
        _one_origin_one_joint(MM, 0.1478, 0.3612, HIP_RIGHT, KNEE_RIGHT); // 14
        _one_origin_one_joint(MM, 0.0481, 0.4352, KNEE_RIGHT, ANKLE_RIGHT); // 15
        _one_origin_one_joint(MM, 0.0129, (4. / 3.) * (0.4014 - 1. / 4.), ANKLE_RIGHT, FOOT_RIGHT); // 16
        break;
    case AVERAGE:
        _one_origin_two_joints(MM, 0.1182, 0.55175, SPINE_NAVEL, HIP_LEFT, HIP_RIGHT); // 1
        _one_origin_one_joint(MM, 0.1549, 0.4507, SPINE_CHEST, SPINE_NAVEL); // 2
        _two_origins_one_joint(MM, 0.15705, 0.2538, CLAVICLE_LEFT, CLAVICLE_RIGHT, SPINE_CHEST); // 3
        _one_origin_two_joints(MM, 0.0681, 0.66666667, HEAD, EAR_LEFT, EAR_RIGHT); // 4
        _one_origin_one_joint(MM, 0.0263, 0.5763, SHOULDER_LEFT, ELBOW_LEFT); // 5
        _one_origin_one_joint(MM, 0.015, 0.45665, ELBOW_LEFT, WRIST_LEFT); // 6
        _one_origin_one_joint(MM, 0.00585, 0.7687, WRIST_LEFT, HAND_LEFT); // 7
        _one_origin_one_joint(MM, 0.0263, 0.5763, SHOULDER_RIGHT, ELBOW_RIGHT); // 8
        _one_origin_one_joint(MM, 0.015, 0.45665, ELBOW_RIGHT, WRIST_RIGHT); // 9
        _one_origin_one_joint(MM, 0.00585, 0.7687, WRIST_RIGHT, HAND_RIGHT); // 10
        _one_origin_one_joint(MM, 0.1447, 0.38535, HIP_LEFT, KNEE_LEFT); // 11
        _one_origin_one_joint(MM, 0.0457, 0.43735, KNEE_LEFT, ANKLE_LEFT); // 12
        _one_origin_one_joint(MM, 0.0133, 0.2286, ANKLE_LEFT, FOOT_LEFT); // 13
        _one_origin_one_joint(MM, 0.1447, 0.38535, HIP_RIGHT, KNEE_RIGHT); // 14
        _one_origin_one_joint(MM, 0.0457, 0.43735, KNEE_RIGHT, ANKLE_RIGHT); // 15
        _one_origin_one_joint(MM, 0.0133, 0.2286, ANKLE_RIGHT, FOOT_RIGHT); // 16
        break;
    default:
        std::cout << "Invalid Branch" << std::endl;
        break;
    }
    return MM;
}
