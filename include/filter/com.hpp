#pragma once
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <filter/com.hpp>
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

    std::tuple<Point<Value>, Point<Value>> into_center_and_normal()
    {
        Eigen::Vector3f va(a.x, a.y, a.z);
        Eigen::Vector3f vb(b.x, b.y, b.z);
        Eigen::Vector3f vc(c.x, c.y, c.z);
        Eigen::Vector3f vd(d.x, d.y, d.z);

        auto vml = va + 0.5 * (vd - va);
        auto vmr = vb + 0.5 * (vc - vb);

        // 1000 = Convert from mm into m
        auto vcenter = (vml + 0.5 * (vmr - vml)) / 1000;

        auto vnorm = (vc - vcenter).cross(vd - vcenter) / 1000;

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

template <typename Value>
class SkeletonStabilityMetrics {
public:
    std::vector<Point<Value>> m_filtered_positions = std::vector<Point<Value>>();
    std::vector<Point<Value>> m_filtered_velocities = std::vector<Point<Value>>();
    MatrixXd m_MM;

    SkeletonStabilityMetrics(MatrixXd mm)
        : m_MM(mm)
    {
    }

    Point<Value> calculate_com()
    {
        Point<Value> com(0.0, 0.0, 0.0);
        for (int joint = 0; joint < this->joint_count(); ++joint) {
            com.x += m_filtered_positions[joint].x * m_MM(0, joint);
            com.y += m_filtered_positions[joint].y * m_MM(0, joint);
            com.z += m_filtered_positions[joint].z * m_MM(0, joint);
        }
        return com;
    }

    Point<Value> calculate_com_dot()
    {
        Point<Value> com(0.0, 0.0, 0.0);
        for (int joint = 0; joint < this->joint_count(); ++joint) {
            com.x += m_filtered_velocities[joint].x * m_MM(0, joint);
            com.y += m_filtered_velocities[joint].y * m_MM(0, joint);
            com.z += m_filtered_velocities[joint].z * m_MM(0, joint);
        }
        return com;
    }

    Point<Value> calculate_x_com(
        Value l // length of inverted pendelum
    )
    {
        auto com = calculate_com_dot();
        auto com_dot = calculate_com_dot();

        Value g = 9.81; // m/s
        Value w_0 = g / l;
        Point<Value> x_com(0.0, 0.0, 0.0);
        x_com.x = com.x + (com_dot.x / w_0);
        x_com.y = com.y + (com_dot.y / w_0);
        x_com.z = com.z + (com_dot.z / w_0);
        return x_com;
    }

    void store_step(std::vector<Point<Value>> positions, std::vector<Point<Value>> velocities)
    {
        m_filtered_positions = positions;
        m_filtered_velocities = velocities;
    }
};
