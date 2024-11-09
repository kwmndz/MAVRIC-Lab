#ifndef DYNAMICBAYESIANFILTERING_H
#define DYNAMICBAYESIANFILTERING_H

#include <vector>
#include <tuple>
#include <string>

struct SensorData {
    int occupied_points;
    int total_points;
    double recognized_area; // RA(X_t)
    double total_area; // AOI

    friend std::ostream& operator<<(std::ostream& os, const SensorData& data);
};

class DBF {
public:
    DBF(double threshold = 0.85);
    double initialize_belief(int group_a_points);
    double state_transition_probability(const SensorData& sensor_data);
    double observation_likelihood(const SensorData& sensor_data);
    double normalize(double belief);
    std::tuple<double, bool, std::string> update_belief(const SensorData& sensor_data);
    
private:
    double belief_prior;
    double threshold;
};

int steps_to_local_min(const std::vector<double>& pos_c, const std::vector<double>& min_predicted, double step_size = 0.1);

#endif // DYNAMICBAYESIANFILTERING_H