#include <random>
#include <iostream>
#include <filter/PointFilter3D.hpp>
#include <filter/GenericFilter1D.hpp>
#include <filter/SimplePointFilter3D.hpp>
#include <filter/SimpleGenericFilter1D.hpp>
#include <matplotlibcpp/matplotlibcpp.h>

namespace plt = matplotlibcpp;

double simulate_sin(double start, double velocity, double time, std::normal_distribution<double>& a, std::mt19937& gen) {
    return start + (velocity * time) + 6*std::sin((time/20)*std::numbers::pi) + std::pow(a(gen), 2) / 2.0;
}
double simulate(double start, double velocity, double time, std::normal_distribution<double>& a, std::mt19937& gen) {
    return start + (velocity * time) + std::pow(a(gen), 2) / 2.0;
}
double add_noise(double value, std::normal_distribution<double>& d, std::mt19937& gen) {
    double noise = d(gen);
    std::cout << "noise: " << noise << std::endl;
    return value + d(gen);
}

int main() {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution d{0.0, 2.0};

    std::normal_distribution a{0.0, 1.1};


    double measurement_error = 14.0;
    double process_error = 0.52;
    double threshold = 10e9;

    auto [sad, sc, sg] = let_simple_matricies_appear_magically();
    auto s_filter = SimpleGenericFilter1D<double>(sad, sad, sc, sg, sg, measurement_error, process_error, threshold);

    auto [ad, c, g] = let_matricies_appear_magically();
    auto filter = GenericFilter1D<double>(ad, ad, c, g, g, measurement_error, process_error, threshold);
    double start = 5;
    double velocity = 1;
    double delta_t = 0.5;

    auto s_initial_state = MatrixXd(2, 1);
    s_initial_state(0, 0) = start;
    s_initial_state(1, 0) = 0;
    auto s_initial_errors = MatrixXd::Identity(2, 2);
    s_filter.init(s_initial_state, s_initial_errors);

    auto initial_state = MatrixXd(3, 1);
    initial_state(0, 0) = start;
    initial_state(1, 0) = 0;
    initial_state(2, 0) = 0;
    auto initial_errors = MatrixXd::Identity(3, 3);
    filter.init(initial_state, initial_errors);

    std::vector<double> timestamps;

    std::vector<double> simulation;
    std::vector<double> measurements;

    std::vector<double> predictions;
    std::vector<double> filtered_positions;
    std::vector<double> filtered_velocities;

    std::vector<double> s_predictions;
    std::vector<double> s_filtered_positions;
    std::vector<double> s_filtered_velocities;

    for (int i = 0; i < 100; ++i) {
        double sim;
        if (i < 50) {
            sim = simulate_sin(start, velocity, i * delta_t, a, gen);
        } else {
            sim = simulate(start, velocity, i * delta_t, a, gen);
        }
        double measurement = add_noise(sim, d, gen);
        auto [position, velocity, prediction] = filter.step_(measurement, delta_t);
        auto [s_position, s_velocity, s_prediction] = s_filter.step_(measurement, delta_t);

        timestamps.push_back(i*delta_t);
        simulation.push_back(sim);
        measurements.push_back(measurement);

        predictions.push_back(prediction);
        filtered_positions.push_back(position);
        filtered_velocities.push_back(velocity);

        s_predictions.push_back(s_prediction);
        s_filtered_positions.push_back(s_position);
        s_filtered_velocities.push_back(s_velocity);
    }

    plt::title("Simulation Measurement and Filtered");
    plt::named_plot("Simulation", timestamps, simulation);
    plt::named_plot("Measurements", timestamps, measurements);
    plt::named_plot("Filtered", timestamps, filtered_positions);
    plt::named_plot("Predictions", timestamps, predictions);
    plt::named_plot("S Filtered", timestamps, s_filtered_positions);
    plt::legend();
    plt::show(true);
    plt::cla();

}
