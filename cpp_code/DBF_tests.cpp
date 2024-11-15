#include "DynamicBayesianFiltering.hpp"
#include <cassert>
#include <vector>
#include <iostream>

//g++ -O3 DBF_tests.cpp DynamicBayesianFiltering.cpp -o test.exe

void test_initialize_belief() {
    DBF dbf;
    double belief = dbf.initialize_belief(10);
    assert(std::abs(belief - 0.1) < 1e-5);
}

void test_state_transition_probability() {
    DBF dbf;
    SensorData sensor_data = {5, 10, 0.0, 0.0};
    double transition_prob = dbf.state_transition_probability(sensor_data);
    assert(std::abs(transition_prob - 1.5) < 1e-5);
}

void test_observation_likelihood() {
    DBF dbf;
    SensorData sensor_data = {0, 0, 20.0, 100.0};
    double likelihood = dbf.observation_likelihood(sensor_data);
    assert(std::abs(likelihood - 0.2) < 1e-5);
}

void test_normalize() {
    DBF dbf;
    double normalized_belief = dbf.normalize(1.5);
    assert(normalized_belief == 1.0);

    normalized_belief = dbf.normalize(-0.5);
    assert(normalized_belief == 0.0);

    normalized_belief = dbf.normalize(0.5);
    assert(normalized_belief == 0.5);
}

void test_update_belief() {
    DBF dbf;
    dbf.initialize_belief(10);
    SensorData sensor_data = {5, 10, 20.0, 100.0};
    auto [belief, is_confident, log_str] = dbf.update_belief(sensor_data);

    std::cout << log_str << std::endl;
    //assert(std::abs(belief - 0.3) < 1e-5);
    assert(is_confident == false);
}

void test_steps_to_local_min() {
    std::vector<double> pos_c = {0.0, 0.0};
    std::vector<double> min_predicted = {1.0, 1.0};
    int steps = steps_to_local_min(pos_c, min_predicted);
    assert(steps == 15); // sqrt(2) / 0.1 rounded up is approx 15
}

int main() {
    test_initialize_belief();
    test_state_transition_probability();
    test_observation_likelihood();
    test_normalize();
    test_update_belief();
    test_steps_to_local_min();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}