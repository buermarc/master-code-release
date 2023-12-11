#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

using Eigen::MatrixXd;

template <typename Value>
class GenericFilter1D {
    Value value;
    MatrixXd corrected_state; // 2x1
    MatrixXd corrected_errors; // 2x1
    MatrixXd Ad;
    MatrixXd B;
    MatrixXd G;
    MatrixXd Q; // Q(k) = Var(z(k))

    MatrixXd C;
    MatrixXd CT;

    Value measurement_noise;
    Value system_noise;
    Value threshold;

public:
    GenericFilter1D() {};

    GenericFilter1D(MatrixXd ad, MatrixXd b, MatrixXd c, MatrixXd g, MatrixXd q,
        Value m_measurement_noise, Value m_system_noise,
        Value m_threshold)
    {
        Ad = ad;
        B = b;
        C = c;
        G = g;
        Q = q;
        measurement_noise = std::pow(m_measurement_noise, 2);
        system_noise = m_system_noise;
        threshold = m_threshold;

        CT = C.transpose();
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
    }

    std::tuple<Value, Value> step(Value value, Value time_diff)
    {
        // Hard coded sub A = [1, Ts; 0, 1]
        MatrixXd Adn = Ad.replicate(1, 1);
        Adn(0, 1) = time_diff;

        // Again: hard coded sub G = [Ts^2/2; Ts]
        MatrixXd Gdn = G.replicate(1, 1);
        Gdn(0, 0) = std::pow(time_diff, 2) / 2;
        Gdn(1, 0) = time_diff;

        auto AdnT = Adn.transpose();
        auto GdnT = Gdn.transpose();

        std::cout << "Adn" << std::endl;
        std::cout << Adn << std::endl;
        std::cout << std::endl;

        std::cout << "Gdn" << std::endl;
        std::cout << Gdn << std::endl;
        std::cout << std::endl;

        MatrixXd predicted_state = Adn * corrected_state;
        std::cout << "predicted_state" << std::endl;
        // std::cout << predicted_state << std::endl;
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

        MatrixXd measurement_noise_matrix(1, 1);
        measurement_noise_matrix(0, 0) = measurement_noise;
        if (std::abs(innovation_norm) < threshold) {
            // std::cout << "use correction" << std::endl;
            //  We are recalculating tmp but that is fine as it still a bit
            //  different as we are using matrices here
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
