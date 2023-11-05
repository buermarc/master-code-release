#pragma once
#include <cstddef>
#include <functional>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "../Point.hpp"
#include "../Utils.hpp"

using Eigen::MatrixXd;
using Eigen::seq;

template <typename Value>
class AdaptiveRigidJointConstructFilter3 {

    MatrixXd corrected_projected_state;
    MatrixXd corrected_projected_errors;

    MatrixXd Ad;
    MatrixXd Gd;
    // MatrixXd Q; // Q(k) = Var(z(k))

    MatrixXd C;
    MatrixXd CT;

    MatrixXd phi_1;
    MatrixXd phi_2;

    // TODO: should this really be of type Value
    // Expects 9x1 matrix, [p1_x, p1_y, p1_z, p2_x, ..., p3_z]
    MatrixXd measurement_noise;
    MatrixXd system_noise;
    Value threshold;
    std::function<MatrixXd(MatrixXd, Value)> sub_ad;
    std::function<MatrixXd(MatrixXd, Value)> sub_gd;

    std::vector<int> joints;

public:
    std::vector<int> get_joints()
    {
        return joints;
    }

    // measurement_errors => var
    static AdaptiveRigidJointConstructFilter3<Value> default_init(std::vector<int> joints, MatrixXd measurement_errors)
    {
        auto eye = [](int size) { return MatrixXd::Identity(size, size); };
        auto zero = [](int size) { return MatrixXd::Zero(size, size); };

        MatrixXd Al(18, 18);
        MatrixXd first_al(9, 18);
        MatrixXd second_al(9, 18);

        first_al << zero(9), eye(9);
        second_al << zero(9), zero(9);

        Al << first_al, second_al;

        MatrixXd Cl(9, 18);
        MatrixXd Gl(18, 9);
        // Gld = [(Ts^2/2)*eye(9,9); Ts*eye(9,9)]; % considering that acceleration has normal ditribution with zero mean, See truck example in wikipedia and Machthaler und Dingler (2017)

        Cl << eye(9), zero(9);
        Gl << eye(9), eye(9);

        MatrixXd phi_1(9, 9);
        MatrixXd phi_2(9, 9);

        phi_1 << eye(3), -1 * eye(3), zero(3), -1 * eye(3), eye(3), zero(3), zero(3), zero(3), zero(3);
        phi_2 << zero(3), zero(3), zero(3), zero(3), eye(3), -1 * eye(3), zero(3), -1 * eye(3), eye(3);

        MatrixXd m_measurement_noise = zero(3);
        {
            int i = 0;
            for (auto element : joints) {
                m_measurement_noise.row(i) = measurement_errors.row(element);
                ++i;
            }
        } // remove i from leaking into outer scope

        Value factor_system_noise = 1.0 / 3;
        Value vmax = 10.0 * factor_system_noise;
        Value sigma_system_noise = vmax / 3;
        MatrixXd system_noise = eye(9) * std::pow(sigma_system_noise, 2);

        Value threshold = 5;

        auto sub_ad = [](MatrixXd Ad, Value time_diff) {
            MatrixXd result(18, 18);
            MatrixXd first = Ad.array() * time_diff;
            MatrixXd second = ((1 - Ad.array()) * Ad.exp().array());
            result = first + second;
            return result;
        };

        auto sub_gd = [](MatrixXd Gd, Value time_diff) {
            MatrixXd result(18, 9);
            MatrixXd first = MatrixXd::Identity(9, 9) * (std::pow(time_diff, 2) / 2);
            MatrixXd second = MatrixXd::Identity(9, 9) * time_diff;
            result << first, second;
            return result;
        };

        return AdaptiveRigidJointConstructFilter3<Value>(
            Al,
            Cl,
            Gl,
            phi_1,
            phi_2,
            m_measurement_noise,
            system_noise,
            threshold,
            sub_ad,
            sub_gd,
            joints);
    }

    AdaptiveRigidJointConstructFilter3(
        MatrixXd ad,
        MatrixXd c,
        MatrixXd gd,
        MatrixXd m_phi_1,
        MatrixXd m_phi_2,
        MatrixXd m_measurement_noise,
        MatrixXd m_system_noise,
        Value m_threshold,
        std::function<MatrixXd(MatrixXd, Value)> m_sub_ad,
        std::function<MatrixXd(MatrixXd, Value)> m_sub_gd,
        std::vector<int> m_joints)
    {
        Ad = ad;
        C = c;
        Gd = gd;
        phi_1 = m_phi_1;
        phi_2 = m_phi_2;

        MatrixXd reshaped = m_measurement_noise.reshaped<Eigen::RowMajor>(9, 1);
        auto sqrt = reshaped.array().sqrt();
        auto sqrt_10 = 10 * sqrt.array();
        MatrixXd pow_measurement_noise = sqrt_10.array().pow(2);
        measurement_noise = pow_measurement_noise.asDiagonal();

        system_noise = m_system_noise;
        threshold = m_threshold;
        sub_ad = m_sub_ad;
        sub_gd = m_sub_gd;

        CT = C.transpose();
        joints = m_joints;
    };

    // what do we need?
    void init(
        MatrixXd inital_state,
        MatrixXd initial_errors)
    {
        corrected_projected_state = inital_state;
        corrected_projected_errors = initial_errors;
    }

    MatrixXd step(MatrixXd measurement, Value time_diff)
    {
        /**
         * We make a step for **ONE** rigid joint construct, meaning a
         * combination of point_a, point_b, and point_c.
         *
         * We assume that however provides us with the measurement vector
         * provides the correct measurements for the joints.
         */
        // measurement should be ? 3x1 mxn rowxcol
        auto eye = [](int size) { return MatrixXd::Identity(size, size); };
        auto zero = [](int size) { return MatrixXd::Zero(size, size); };

        MatrixXd Adn;
        Adn = sub_ad(Ad, time_diff);

        MatrixXd Gdn;
        Gdn = sub_gd(Gd, time_diff);

        auto AdnT = Adn.transpose();
        auto GdnT = Gdn.transpose();

        /// Prediction step
        MatrixXd predicted_state = Adn * corrected_projected_state;
        // system_noise => process model noise 9x9
        MatrixXd predicted_errors = Adn * corrected_projected_errors * AdnT + Gdn * system_noise * GdnT;

        /// Correction step:
        /// calculate kalman gain and use to apply the residual
        MatrixXd tmp = C * predicted_errors * CT + measurement_noise;
        auto pseudo_inv = tmp.completeOrthogonalDecomposition().pseudoInverse();
        MatrixXd K_value = predicted_errors * CT * pseudo_inv;
        MatrixXd corrected_state = predicted_state + K_value * (measurement - (C * predicted_state));
        MatrixXd corrected_errors = (eye(18) - K_value * C) * predicted_errors;

        /// Projection step:
        /// Probably something like: project the corrected state and errors
        /// into the wanted solution space that is acecptable

        auto sub_corrected_state_T = corrected_state(seq(0, 8), 0).transpose();
        MatrixXd phi(2, 9); // Should be (2, 9)
        // concat the two calculations
        phi << sub_corrected_state_T * phi_1, sub_corrected_state_T * phi_2;
        MatrixXd phi_T = phi.transpose();

        MatrixXd eye_9 = eye(9);
        MatrixXd zero_9 = zero(9);

        MatrixXd tmp_result(18, 18);

        MatrixXd first(9, 18);
        MatrixXd second(9, 18);
        first << eye_9, zero_9;
        second << zero_9, (eye_9 - (phi_T * (phi * phi_T).inverse() * phi));
        tmp_result << first, second;

        corrected_projected_state = tmp_result * corrected_state;

        corrected_projected_errors = tmp_result * corrected_errors;

        return corrected_projected_state;
    }
};

template <typename Value, typename AdaptivePointFilter>
class AdaptiveConstrainedSkeletonFilter {
    size_t n_joints;
    bool initialized = false;
    Value last_time;

    // std::vector<std::vector<int>> constrained_joint_groups = { { 19, 20, 21 }, { 23, 24, 25 }, { 6, 7, 8 }, { 13, 14, 15 } };
    std::vector<std::vector<int>> constrained_joint_groups = { { 18, 19, 20 }, { 22, 23, 24 }, { 5, 6, 7 }, { 12, 13, 14 } };
    std::unordered_map<int, AdaptiveRigidJointConstructFilter3<Value>> joint_group_filters;
    std::unordered_map<int, AdaptivePointFilter> single_joint_filters;

public:
    int joint_count() { return n_joints; }

    AdaptiveConstrainedSkeletonFilter(
        int m_n_joints,
        MatrixXd measurement_errors)
        : n_joints(m_n_joints)
    {
        for (auto joint_group : constrained_joint_groups) {
            auto filter = AdaptiveRigidJointConstructFilter3<Value>::default_init(joint_group, measurement_errors);
            joint_group_filters.insert(std::make_pair(joint_group.front(), filter));
        }

        // Skip joints which are already covered in constrained joint groups
        std::set<int> flat;
        unroll(constrained_joint_groups, flat);

        for (int i = 0; i < m_n_joints; ++i) {
            if (flat.find(i) == flat.end()) {
                auto filter = AdaptivePointFilter::default_init(i, measurement_errors);
                single_joint_filters.insert(std::make_pair(i, filter));
            }
        }
    }

    void init(std::vector<Point<Value>> initial_points, Value initial_time)
    {
        if (initialized) {
            return;
        }

        for (auto& [_, filter] : joint_group_filters) {
            auto joints = filter.get_joints();
            MatrixXd initial_state(18, 1);
            int i = 0;
            for (auto joint : joints) {
                Point<Value> point = initial_points[joint];
                initial_state(3 * i, 0) = point.x;
                initial_state(3 * i + 1, 0) = point.y;
                initial_state(3 * i + 2, 0) = point.z;
                ++i;
            }
            // Fill velocities with zero
            initial_state << initial_state(seq(0, 8), 0), MatrixXd::Zero(9, 1);
            MatrixXd initial_error = MatrixXd::Identity(18, 18);
            filter.init(initial_state, initial_error);
        }

        for (auto& [i, filter] : single_joint_filters) {
            filter.init(initial_points[i]);
        }

        last_time = initial_time;
        initialized = true;
    }

    bool is_initialized() { return initialized; }

    std::tuple<std::vector<Point<Value>>, std::vector<Point<Value>>> step(std::vector<Point<Value>> values,
        Value new_time)
    {
        std::vector<Point<Value>> positions(32);
        std::vector<Point<Value>> velocities(32);
        std::fill(positions.begin(), positions.end(), Point(0.0, 0.0, 0.0));
        std::fill(velocities.begin(), velocities.end(), Point(0.0, 0.0, 0.0));

        auto time_diff = new_time - last_time;

        for (auto& [_, filter] : joint_group_filters) {
            auto joints = filter.get_joints();
            MatrixXd measurement(9, 1);
            int i = 0;
            for (auto joint : joints) {
                measurement(3 * i, 0) = values[joint].x;
                measurement(3 * i + 1, 0) = values[joint].y;
                measurement(3 * i + 2, 0) = values[joint].z;
                ++i;
            }
            MatrixXd result = filter.step(measurement, time_diff);

            i = 0;

            for (auto joint : joints) {
                positions[joint].x = result(3 * i, 0);
                positions[joint].y = result(3 * i + 1, 0);
                positions[joint].z = result(3 * i + 2, 0);
                ++i;
            }

            i = 0; // Start with offset of 8 in results to skip positions
            for (auto joint : joints) {
                velocities[joint].x = result(8 + 3 * i, 0);
                velocities[joint].y = result(8 + 3 * i + 1, 0);
                velocities[joint].z = result(8 + 3 * i + 2, 0);
                ++i;
            }
        }

        // Skip joints which are already covered in constrained joint groups
        for (auto& [i, filter] : single_joint_filters) {
            auto [position, velocity] = filter.step(values[i], time_diff);
            positions[i] = position;
            velocities[i] = velocity;
        }

        last_time = new_time;
        return std::make_tuple(positions, velocities);
    }

    Point<Value> calculate_com(
        std::vector<Point<Value>> filtered_positions,
        MatrixXd MM) // MM[1x32]
    {
        Point<Value> com(0.0, 0.0, 0.0);
        for (int joint = 0; joint < this->joint_count(); ++joint) {
            com.x += filtered_positions[joint].x * MM(0, joint);
            com.y += filtered_positions[joint].y * MM(0, joint);
            com.z += filtered_positions[joint].z * MM(0, joint);
        }
        return com;
    }

    Point<Value> calculate_x_com(
        Point<Value> com,
        Point<Value> com_dot,
        Value l // length of inverted pendelum
    )
    {
        Value g = 9.81; // m/s
        Value w_0 = g / l;
        Point<Value> x_com(0.0, 0.0, 0.0);
        x_com.x = com.x + (com_dot.x / w_0);
        x_com.y = com.y + (com_dot.y / w_0);
        x_com.z = com.z + (com_dot.z / w_0);
        return x_com;
    }
};
