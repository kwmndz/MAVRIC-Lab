#include "DynamicBayesianFiltering.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <sstream>

std::ostream& operator<<(std::ostream& os, const SensorData& data) {
    os << "SensorData(occupied_points=" << data.occupied_points 
       << ", total_points=" << data.total_points 
       << ", recognized_area=" << data.recognized_area 
       << ", total_area=" << data.total_area << ")";
    return os;
}

DBF::DBF(double threshold) : threshold(threshold), belief_prior(0.0) {}

double DBF::initialize_belief(int group_a_points) {
    belief_prior = 1.0 / group_a_points;
    return belief_prior;
}

double DBF::state_transition_probability(const SensorData& sensor_data) {
    double alpha = static_cast<double>(sensor_data.occupied_points) / sensor_data.total_points;
    return alpha + 1.0;
}

double DBF::observation_likelihood(const SensorData& sensor_data) {
    return sensor_data.recognized_area / sensor_data.total_area;
}

double DBF::normalize(double belief) {
    return std::max(0.0, std::min(1.0, belief));
}

std::tuple<double, bool, std::string> DBF::update_belief(const SensorData& sensor_data) {
    if (belief_prior == 0.0) {
        throw std::runtime_error("Remember to initialize the belief first (initialize_belief function)");
    }

    double prediction = state_transition_probability(sensor_data);
    double correction = observation_likelihood(sensor_data);
    double posterior = prediction * correction * belief_prior;

    std::ostringstream log_str;
    log_str << "Prediction: " << prediction 
            << ", Correction: " << correction 
            << ", Prior: " << belief_prior 
            << ", Posterior: " << posterior;

    double belief_current = normalize(posterior);
    belief_prior = belief_current;

    log_str << " Belief after normalization: " << belief_current;
    return {belief_current, belief_current >= threshold, log_str.str()};
}

int steps_to_local_min(const std::vector<double>& pos_c, const std::vector<double>& min_predicted, double step_size) {
    double distance = 0.0;
    for (size_t i = 0; i < pos_c.size(); ++i) {
        distance += std::pow(min_predicted[i] - pos_c[i], 2);
    }
    distance = std::sqrt(distance);
    return static_cast<int>(std::ceil(distance / step_size));
}