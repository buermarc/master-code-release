#pragma once
#include <filter/com.hpp>
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <tuple>

#include "Point.hpp"
using Eigen::MatrixXd;

template <typename Value>
class Plane {
public:
    Point<Value> a, b, c, d;

    Plane(Point<Value> m_a,
        Point<Value> m_b,
        Point<Value> m_c,
        Point<Value> m_d)
        : a(m_a)
        , b(m_b)
        , c(m_c)
        , d(m_d) {};

    std::tuple<Point<Value>, Point<Value>> into_center_and_normal();
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
    int joint_2);

void _two_origins_one_joint(
    MatrixXd& MM,
    double normalized_mass,
    double normalized_com_position,
    int origin_joint_1,
    int origin_joint_2,
    int joint);

void _one_origin_one_joint(
    MatrixXd& MM,
    double normalized_mass,
    double normalized_com_position,
    int origin_joint,
    int joint);

template <typename Value>
Plane<Value> azure_kinect_bos(std::vector<Point<Value>> joints)
{

    auto ankle_left = joints[ANKLE_LEFT];
    auto ankle_right = joints[ANKLE_RIGHT];

    auto foot_left = joints[FOOT_LEFT];
    auto foot_right = joints[FOOT_RIGHT];

    ankle_left.y = foot_left.y;
    ankle_right.y = foot_right.y;

    return Plane(ankle_left, ankle_right, foot_right, foot_left);
}

MatrixXd get_azure_kinect_com_matrix(SEX sex = AVERAGE);
