#pragma once
#include "filter/com.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <functional>
#include <unsupported/Eigen/MatrixFunctions>

using Eigen::MatrixXd;

template <typename Value>
class AdaptiveBarShalomFilter1D {
    Value value;
    MatrixXd corrected_state; // 2x1
    MatrixXd corrected_errors; // 2x1
    MatrixXd Ad;
    MatrixXd Gd;

    MatrixXd C;
    MatrixXd CT;

    Value measurement_noise;
    Value system_noise;
    Value threshold;
    Value system_scale_factor;
    int system_scale_count;
    Value innovation_norm_max;

    std::function<MatrixXd(MatrixXd, Value)> sub_ad;
    std::function<MatrixXd(MatrixXd, Value)> sub_gd;

public:
    static AdaptiveBarShalomFilter1D<Value> default_init(
        Value measurement_error
    ) {
        MatrixXd A(2, 2);
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

        auto sub_ad = [](MatrixXd Ad, Value time_diff) {
            MatrixXd result = Ad.replicate(1, 1);
            result(0, 1) = time_diff;
            return result;
        };

        auto sub_gd = [](MatrixXd Gd, Value time_diff) {
            MatrixXd result(2, 1);
            result(0, 0) = std::pow(time_diff, 2) / 2;
            result(1, 0) = time_diff;
            return result;
        };

        Value measurement_noise = std::sqrt(measurement_error) * 10;
        Value system_noise = std::pow(((1.0 / 3) * 10) / 3, 2);
        Value threshold = 5.0;
        Value system_scale_factor = 100;
        Value innovation_norm_max = 3.0;
        return AdaptiveBarShalomFilter1D(
            A, C, G, measurement_noise, system_noise, threshold, system_scale_factor, innovation_norm_max, sub_ad, sub_gd
        );

    };

    AdaptiveBarShalomFilter1D() {};

    AdaptiveBarShalomFilter1D(MatrixXd ad, MatrixXd c, MatrixXd gd,
        Value m_measurement_noise, Value m_system_noise,
        Value m_threshold, Value m_system_scale_factor, int m_innovation_norm_max,
        std::function<MatrixXd(MatrixXd, Value)> m_sub_ad,
        std::function<MatrixXd(MatrixXd, Value)> m_sub_gd)
    {
        Ad = ad;
        C = c;
        Gd = gd;
        measurement_noise = std::pow(m_measurement_noise, 2);
        system_noise = m_system_noise;
        threshold = m_threshold;

        sub_ad = m_sub_ad;
        sub_gd = m_sub_gd;

        system_scale_factor = m_system_scale_factor;
        CT = C.transpose();
        system_scale_count = 0;
        innovation_norm_max = m_innovation_norm_max;
    };

    void init(MatrixXd initial_state, MatrixXd initial_errors)
    {
        // how do they look and where do they come from
        // look:
        // think was 2x1
        // origin:
        // just the initial value that is recorded
        corrected_state = initial_state;
        corrected_errors = initial_errors;
        // In our case the initial measurement is contained in the initial_state
        Value initial_measurement = initial_state(0, 0);
    }

    std::tuple<Value, Value> step(Value value, Value time_diff)
    {

        MatrixXd Adn;
        Adn = sub_ad(Ad, time_diff);

        MatrixXd Gdn;
        Gdn = sub_gd(Gd, time_diff);

        auto AdnT = Adn.transpose();
        auto GdnT = Gdn.transpose();

        std::cout << "Adn" << std::endl;
        std::cout << Adn << std::endl;
        std::cout << std::endl;

        std::cout << "Gdn" << std::endl;
        std::cout << Gdn << std::endl;
        std::cout << std::endl;

        // Update measurement noise (R)

        MatrixXd predicted_state = Adn * corrected_state;
        std::cout << "predicted_state" << std::endl;
        std::cout << predicted_state << std::endl;
        MatrixXd predicted_errors = Adn * corrected_errors * AdnT + Gdn * system_noise * GdnT;
        std::cout << "predicted_errors" << std::endl;
        std::cout << predicted_errors << std::endl;

        Value innovation = value - (C * predicted_state).array()(0); // residual
        std::cout << "innovation" << std::endl;
        std::cout << innovation << std::endl;
        std::cout << std::endl;
        Value innovation_covariance = (C * predicted_errors * CT).array()(0) + measurement_noise;
        std::cout << "innovation_covariance" << std::endl;
        std::cout << innovation_covariance << std::endl;
        std::cout << std::endl;
        Value sigma_value = std::sqrt(innovation_covariance);
        std::cout << "sigma_value" << std::endl;
        std::cout << sigma_value << std::endl;
        std::cout << std::endl;
        Value innovation_norm = innovation / sigma_value;
        std::cout << "innovation_norm" << std::endl;
        std::cout << innovation_norm << std::endl;
        std::cout << std::endl;
        Value abs_innovation_norm = std::abs(innovation_norm);

        MatrixXd measurement_noise_matrix(1, 1);
        measurement_noise_matrix(0, 0) = measurement_noise;

        if (abs_innovation_norm > innovation_norm_max) {
            system_noise *= system_scale_factor;
            ++system_scale_count;
        } else if (system_scale_count > 0) {
            --system_scale_count;
        }

        if (abs_innovation_norm <= threshold) {
            std::cout << "use correction" << std::endl;
            // We are recalculating tmp but that is fine as it still a bit
            // different as we are using matrices here
            MatrixXd tmp = C * predicted_errors * CT + measurement_noise_matrix;
            // TODO: if we have just one value we could also just use a simple
            // inversion instead of a pseudo inverse
            auto pseudo_inv = tmp.completeOrthogonalDecomposition().pseudoInverse();
            std::cout << "pseudo_inv==1/innovation_covariance" << std::endl;
            std::cout << pseudo_inv << "==" << 1 / innovation_covariance << std::endl;
            MatrixXd K_value = predicted_errors * CT * (1 / innovation_covariance);
            std::cout << "K_value" << std::endl;
            std::cout << K_value << std::endl;
            corrected_state = predicted_state + K_value * innovation;
            std::cout << "corrected_state" << std::endl;
            std::cout << corrected_state << std::endl;
            auto eye = MatrixXd::Identity(2, 2);
            corrected_errors = (eye - K_value * C) * predicted_errors;
            std::cout << "corrected_errors" << std::endl;
            std::cout << corrected_errors << std::endl;
        } else {
            std::cout << "use prediction" << std::endl;
            corrected_state = predicted_state;
            corrected_errors = predicted_errors;
        }
        return std::make_tuple(corrected_state(0, 0), corrected_state(1, 0));
    }
};
