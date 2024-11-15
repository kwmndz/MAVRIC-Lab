#include "Sensor.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include "DynamicBayesianFiltering.hpp"

//g++ -O3 Sensor_tests.cpp DynamicBayesianFiltering.cpp Sensor.cpp -o test.exe 


void test_scan_area() {
    Sensor sensor(10.0, 180.0, 36);
    std::array<double, 3> pos_c = {0.0, 0.0, 0.0};
    std::vector<std::array<double, 3>> obstacles = {{5.0, 5.0, 0.0}, {-5.0, -5.0, 0.0}};
    auto rays = sensor.scan_area(pos_c, obstacles);
    assert(rays.size() == 36);
}

void test_calc_scanned_area() {
    Sensor sensor(10.0, 180.0, 36);
    double area = sensor.calc_scanned_area();
    assert(std::abs(area - (M_PI * 10.0 * 10.0 * 0.5)) < 1e-5);
}

void test_calc_area_of_interest() {
    Sensor sensor(10.0, 180.0, 36);
    std::array<double, 3> pos_c = {0.0, 0.0, 0.0};
    std::array<double, 3> local_min = {10.0, 10.0, 0.0};
    double area_of_interest = sensor.calc_area_of_interest(pos_c, local_min);
    assert(area_of_interest > sensor.calc_scanned_area());
}

void test_get_obstacles_in_recognized_area() {
    Sensor sensor(10.0, 180.0, 36);
    std::array<double, 3> pos_c = {0.0, 0.0, 0.0};
    std::vector<std::array<double, 3>> obstacles = {{5.0, 5.0, 0.0}, {-5.0, -5.0, 0.0}, {15.0, 15.0, 0.0}};
    auto seen_obstacles = sensor.get_obstacles_in_recognized_area(pos_c, obstacles);
    assert(seen_obstacles.size() == 2);
}

void test_get_group_a_points() {
    Sensor sensor(10.0, 180.0, 36);
    std::array<double, 3> pos_c = {0.0, 0.0, 0.0};
    std::array<double, 3> local_min = {10.0, 10.0, 0.0};
    int group_a_points = sensor.get_group_a_points(pos_c, local_min);
    assert(group_a_points > 0);
}

void test_check_for_parallel_forces() {
    Sensor sensor(10.0, 180.0, 36);
    std::array<double, 3> F_att = {1.0, 0.0, 0.0};
    std::array<double, 3> F_rep = {2.0, 0.0, 0.0};
    bool parallel = sensor.check_for_parallel_forces(F_att, F_rep);
    assert(parallel);
}

void test_get_sensor_data() {
    Sensor sensor(10.0, 180.0, 36);
    std::array<double, 3> pos_c = {0.0, 0.0, 0.0};
    std::array<double, 3> local_min = {10.0, 10.0, 0.0};
    std::vector<std::array<double, 3>> obstacles = {{5.0, 5.0, 0.0}, {-5.0, -5.0, 0.0}};
    std::string log_str;
    SensorData data = sensor.get_sensor_data(pos_c, local_min, obstacles, log_str);
    assert(data.occupied_points > 0);
    assert(!log_str.empty());
}

int main() {
    test_scan_area();
    test_calc_scanned_area();
    test_calc_area_of_interest();
    test_get_obstacles_in_recognized_area();
    test_get_group_a_points();
    test_check_for_parallel_forces();
    test_get_sensor_data();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}