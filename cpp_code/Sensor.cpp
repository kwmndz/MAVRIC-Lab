#include "Sensor.hpp"
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <sstream>
#include <iostream>
#include "DynamicBayesianFiltering.hpp"

// Constructor
Sensor::Sensor(double sensor_radius, double sensor_angle, int num_rays)
    : sensor_radius(sensor_radius), sensor_angle(sensor_angle), num_rays(num_rays) {}

// Check for collisions along a path
bool Sensor::check_collision(const std::vector<std::array<double, 3>>& obstacles, const std::array<double, 2>& end_point, const std::array<double, 3>& pos_c, double collision_threshold) {
    int num_steps = static_cast<int>(std::sqrt(std::pow(end_point[0] - pos_c[0], 2) + std::pow(end_point[1] - pos_c[1], 2)));
    for (int step = 1; step <= num_steps; ++step) {
        std::array<double, 2> point_c = { pos_c[0] + (end_point[0] - pos_c[0]) * (step / static_cast<double>(num_steps)),
                                          pos_c[1] + (end_point[1] - pos_c[1]) * (step / static_cast<double>(num_steps)) };
        for (const auto& obs : obstacles) {
            if (std::sqrt(std::pow(point_c[0] - obs[0], 2) + std::pow(point_c[1] - obs[1], 2)) < collision_threshold) {
                return true;
            }
        }
    }
    return false;
}

// Scan the area and detect obstacles
std::vector<std::pair<std::array<double, 2>, bool>> Sensor::scan_area(const std::array<double, 3>& pos_c, const std::vector<std::array<double, 3>>& obstacles) {
    double angle_increment = sensor_angle / num_rays;
    std::vector<std::pair<std::array<double, 2>, bool>> rays;

    for (int i = 0; i < num_rays; ++i) {
        double angle = -sensor_angle / 2 + i * angle_increment;
        std::array<double, 2> direction = { std::cos(angle), std::sin(angle) };
        std::array<double, 2> end_point = { pos_c[0] + direction[0] * sensor_radius, pos_c[1] + direction[1] * sensor_radius };
        bool collision = check_collision(obstacles, end_point, pos_c);
        rays.push_back({ end_point, collision });
    }

    return rays;
}

// Calculate the scanned area
double Sensor::calc_scanned_area() {
    return M_PI * std::pow(sensor_radius, 2) * (sensor_angle / 360.0);
}

// Calculate the area of interest
double Sensor::calc_area_of_interest(const std::array<double, 3>& pos_c, const std::array<double, 3>& local_min) {
    double distance = std::sqrt(std::pow(local_min[0] - pos_c[0], 2) + std::pow(local_min[1] - pos_c[1], 2));
    double current_area = calc_scanned_area();
    double area_from_local_min = M_PI * std::pow(distance, 2) * (sensor_angle / 360.0);
    return current_area + area_from_local_min;
}

// Get obstacles within the sensor's recognized area
std::vector<std::array<double, 3>> Sensor::get_obstacles_in_recognized_area(const std::array<double, 3>& pos_c, const std::vector<std::array<double, 3>>& obstacles) {
    std::vector<std::array<double, 3>> seen_obstacles;
    for (const auto& obs : obstacles) {
        double distance = std::sqrt(std::pow(obs[0] - pos_c[0], 2) + std::pow(obs[1] - pos_c[1], 2));
        if (distance < sensor_radius) {
            seen_obstacles.push_back(obs);
        }
    }
    return seen_obstacles;
}

// Get the number of points in group A
int Sensor::get_group_a_points(const std::array<double, 3>& pos_c, const std::array<double, 3>& local_min, double step_size) {
    double distance = std::sqrt(std::pow(local_min[0] - pos_c[0], 2) + std::pow(local_min[1] - pos_c[1], 2));
    return static_cast<int>(std::ceil(distance / step_size));
}

// Check if forces are parallel
bool Sensor::check_for_parallel_forces(const std::array<double, 3>& F_att, const std::array<double, 3>& F_rep) {
    std::array<double, 3> cross_product = { F_att[1] * F_rep[2] - F_att[2] * F_rep[1],
                                            F_att[2] * F_rep[0] - F_att[0] * F_rep[2],
                                            F_att[0] * F_rep[1] - F_att[1] * F_rep[0] };
    return std::all_of(cross_product.begin(), cross_product.end(), [](double val) { return std::abs(val) < 1e-5; });
}

// Get sensor data and log it
SensorData Sensor::get_sensor_data(const std::array<double, 3>& pos_c, const std::array<double, 3>& local_min, const std::vector<std::array<double, 3>>& obstacles, std::string& log_str) {
    auto rays = scan_area(pos_c, obstacles);
    int num_collisions = std::count_if(rays.begin(), rays.end(), [](const auto& ray) { return ray.second; });
    double recognized_area = calc_scanned_area();
    double total_area = calc_area_of_interest(pos_c, local_min);

    std::ostringstream oss;
    oss << "Num Collisions: " << num_collisions << ", Recognized Area: " << recognized_area << ", Total Area: " << total_area;
    log_str = oss.str();

    return { num_collisions, static_cast<int>(rays.size()), recognized_area, total_area };
}