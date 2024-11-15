#ifndef SENSOR_HPP
#define SENSOR_HPP

#include <vector>
#include <array>
#include <string>

#include "DynamicBayesianFiltering.hpp"

/**
 * @class Sensor
 * @brief A class to represent a sensor that scans an area and detects obstacles.
 * 
 * The Sensor class provides methods to scan an area, calculate scanned areas, 
 * determine areas of interest, and detect obstacles within a recognized area.
 * It also includes methods to check for parallel forces and retrieve sensor data.
 */
class Sensor {
public:
    /**
     * @brief Construct a new Sensor object.
     * 
     * @param sensor_radius The radius of the sensor.
     * @param sensor_angle The angle of the sensor.
     * @param num_rays The number of rays emitted by the sensor.
     */
    Sensor(double sensor_radius, double sensor_angle, int num_rays);

    /**
     * @brief Scan the area around a given position and detect obstacles.
     * 
     * @param pos_c The current position of the sensor.
     * @param obstacles A vector of obstacles represented by their positions.
     * @return A vector of pairs containing the end points of the rays and a boolean indicating if an obstacle was detected.
     */
    std::vector<std::pair<std::array<double, 2>, bool>> scan_area(const std::array<double, 3>& pos_c, const std::vector<std::array<double, 3>>& obstacles);

    /**
     * @brief Calculate the total area scanned by the sensor.
     * 
     * @return The total scanned area.
     */
    double calc_scanned_area();

    /**
     * @brief Calculate the area of interest based on the current position and a local minimum.
     * 
     * @param pos_c The current position of the sensor.
     * @param local_min The local minimum position.
     * @return The area of interest.
     */
    double calc_area_of_interest(const std::array<double, 3>& pos_c, const std::array<double, 3>& local_min);

    /**
     * @brief Get the obstacles within the recognized area of the sensor.
     * 
     * @param pos_c The current position of the sensor.
     * @param obstacles A vector of obstacles represented by their positions.
     * @return A vector of obstacles within the recognized area.
     */
    std::vector<std::array<double, 3>> get_obstacles_in_recognized_area(const std::array<double, 3>& pos_c, const std::vector<std::array<double, 3>>& obstacles);

    /**
     * @brief Get the number of group A points based on the current position and a local minimum.
     * 
     * @param pos_c The current position of the sensor.
     * @param local_min The local minimum position.
     * @param step_size The step size for calculating group A points.
     * @return The number of group A points.
     */
    int get_group_a_points(const std::array<double, 3>& pos_c, const std::array<double, 3>& local_min, double step_size = 0.05);

    /**
     * @brief Check if the attractive and repulsive forces are parallel.
     * 
     * @param F_att The attractive force.
     * @param F_rep The repulsive force.
     * @return True if the forces are parallel, false otherwise.
     */
    bool check_for_parallel_forces(const std::array<double, 3>& F_att, const std::array<double, 3>& F_rep);

    /**
     * @brief Get the sensor data based on the current position, local minimum, and obstacles.
     * 
     * @param pos_c The current position of the sensor.
     * @param local_min The local minimum position.
     * @param obstacles A vector of obstacles represented by their positions.
     * @param log_str A string to store the log information.
     * @return The sensor data.
     */
    SensorData get_sensor_data(const std::array<double, 3>& pos_c, const std::array<double, 3>& local_min, const std::vector<std::array<double, 3>>& obstacles, std::string& log_str);

private:
    double sensor_radius; ///< The radius of the sensor.
    double sensor_angle; ///< The angle of the sensor.
    int num_rays; ///< The number of rays emitted by the sensor.

    /**
     * @brief Check if there is a collision with any obstacle.
     * 
     * @param obstacles A vector of obstacles represented by their positions.
     * @param end_point The end point of the ray.
     * @param pos_c The current position of the sensor.
     * @param collision_threshold The threshold distance for detecting a collision.
     * @return True if a collision is detected, false otherwise.
     */
    bool check_collision(const std::vector<std::array<double, 3>>& obstacles, const std::array<double, 2>& end_point, const std::array<double, 3>& pos_c, double collision_threshold = 1.0);
};

#endif // SENSOR_HPP