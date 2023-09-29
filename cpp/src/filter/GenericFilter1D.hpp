#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using Eigen::MatrixXd;

/**
 * how should the class look like:
 * could do it specific for 3d
 * could also do it just for any value
 *
 * If we are not generic then we can set Adn and the rest of the matrices to a
 * fixed value if we want to be generic this would be a bit trickier
 *
 * Maybe Generic class wich takes the stuff and actual clas fo rour case where
 * we set the stuff in the constructor could also do it just for any value
 *
 * Basically the gated 3d case just sets 3 GenericFilters for each var and then
 * steps with the 3d thing
 *
 *
 * Intgeration things might be the stuff that a specific class needs to
 * implement, because I do not want to have to implement symbolic integration
 */
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

        MatrixXd predicted_state = Adn * corrected_state;
        MatrixXd predicted_errors = Adn * corrected_errors * AdnT + Gdn * system_noise * GdnT;

        auto comb = C * predicted_state;
        Value innovation = value - (C * predicted_state).array()(0);
        Value innovation_covariance = (C * Adn * predicted_errors * AdnT * CT).array()(0) + (C * system_noise * CT).array()(0) + measurement_noise;
        Value sigma_value = std::sqrt(innovation_covariance);
        Value innovation_norm = innovation / sigma_value;

        MatrixXd measurement_noise_matrix(1, 1);
        measurement_noise_matrix(0, 0) = measurement_noise;
        if (innovation_norm < threshold) {
            MatrixXd tmp = C * predicted_errors * CT + measurement_noise_matrix;
            auto pseudo_inv = tmp.completeOrthogonalDecomposition().pseudoInverse();
            MatrixXd K_value = predicted_errors * CT * pseudo_inv;
            corrected_state = predicted_state + K_value * (value - (C * predicted_state).array()(0));
            auto eye = MatrixXd::Identity(2, 2);
            corrected_errors = (eye - K_value * C) * predicted_errors;
        } else {
            corrected_state = predicted_state;
            corrected_errors = predicted_errors;
        }
        return std::make_tuple(corrected_state(0, 0), corrected_state(0, 1));
    }
};
