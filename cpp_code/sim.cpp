#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <filesystem>
#include <thread>
#include <future>
#include "Sensor.hpp" // Assuming Sensor class is defined here
#include "DynamicBayesianFiltering.hpp" // Assuming DBF class and related functions are defined here

using namespace std;
using namespace Eigen;

const double K_ATT = 1.0;  // Attractive constant
const double K_REP = 0.5;  // Repulsive constant
const double D_SAFE = 5.0; // Safe distance / Scanner Radius
const double STEP_SIZE = 0.01;
const double MAX_VEL = 1.0; // Max velocity of UGV

// Attractive force calculation
Vector3d att_force(const Vector3d& ugv_pos, const Vector3d& goal_pos) {
    return K_ATT * (goal_pos - ugv_pos);
}

// Optimized repulsive force calculation
Vector3d rep_force_optimized(const Vector3d& ugv_pos, const vector<Vector3d>& obstacles) {
    Vector3d total_force = Vector3d::Zero();
    for (const auto& obs : obstacles) {
        Vector3d displacement = obs - ugv_pos;
        double distance = displacement.norm();
        if (distance < D_SAFE) {
            Vector3d direction = -displacement / (distance + 1e-25);
            total_force += K_REP * ((1.0 / (distance + 1e-25) - 1.0 / D_SAFE) * (1.0 / pow(distance + 1e-25, 2))) * direction;
        }
    }
    return total_force;
}

// Check if UGV is at a local minimum
bool is_local_min(const Vector3d& force_net, double threshold = 1e-3) {
    return force_net.norm() < threshold;
}

// Calculate force difference
double force_difference(const Vector3d& ugv_pos, const Vector3d& goal_pos, const vector<Vector3d>& obstacles) {
    Vector3d F_att = att_force(ugv_pos, goal_pos);
    if (F_att.norm() < 1e-6) {
        return 10.0;
    }
    Vector3d F_rep = rep_force_optimized(ugv_pos, obstacles);
    return (F_att + F_rep).norm();
}

// Find potential local minimum using Nelder-Mead method
Vector3d find_potential_local_min(const Vector3d& ugv_pos, const Vector3d& goal_pos, const vector<Vector3d>& obstacles, Vector3d guess_i, int max_iter = 1000, double threshold = 1e-8) {
    // Placeholder for optimization logic, here we use a simple approach
    Vector3d local_min = guess_i;
    double min_force_diff = force_difference(guess_i, goal_pos, obstacles);
    for (int i = 0; i < max_iter; ++i) {
        Vector3d new_guess = local_min + Vector3d::Random() * 0.1; // Random small perturbation
        double new_force_diff = force_difference(new_guess, goal_pos, obstacles);
        if (new_force_diff < min_force_diff) {
            local_min = new_guess;
            min_force_diff = new_force_diff;
        }
        if (min_force_diff < threshold) break;
    }
    return local_min;
}

// Generate clustered obstacles
vector<Vector3d> generate_obstacles(int num_obstacles, double step_size = 0.001, pair<int, int> length_range = {5, 20}, array<double, 4> area_size = {100, 100, 0, 0}) {
    vector<Vector3d> obstacles;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);

    Vector2d direction_vector = Vector2d(area_size[0] - area_size[2], area_size[1] - area_size[3]).normalized();
    double tunnel_width = 12.5;
    Vector2d perpendicular_vector(-direction_vector.y(), direction_vector.x());

    for (int i = 0; i < num_obstacles; ++i) {
        double distance_along_tunnel = dis(gen) * sqrt(pow(area_size[0] - area_size[2], 2) + pow(area_size[1] - area_size[3], 2));
        double start_x = area_size[2] + distance_along_tunnel * direction_vector.x();
        double start_y = area_size[3] + distance_along_tunnel * direction_vector.y();
        double width_offset = (dis(gen) - 0.5) * tunnel_width;

        start_x += width_offset * perpendicular_vector.x();
        start_y += width_offset * perpendicular_vector.y();

        start_x = clamp(start_x, area_size[2], area_size[0]);
        start_y = clamp(start_y, area_size[3], area_size[1]);

        double angle = dis(gen) * 2 * M_PI;
        double length = dis(gen) * (length_range.second - length_range.first) + length_range.first;
        int num_points = static_cast<int>(length / step_size) + 1;

        for (int j = 0; j < num_points; ++j) {
            double offset = step_size * j;
            double x = start_x + offset * cos(angle);
            double y = start_y + offset * sin(angle);
            obstacles.emplace_back(Vector3d(x, y, 0));
        }
    }
    return obstacles;
}

// Simulate movement with Dynamic Bayesian Filtering
vector<Vector3d> sim_movement_with_DBF(const Vector3d& pos_i, const Vector3d& goal_pos, const vector<Vector3d>& obstacles, int num_steps) {
    vector<Vector3d> ugv_pos = { pos_i };
    Sensor sensor(D_SAFE, 180, 1000);
    DBF dbf;
    Vector3d estimated_local_min, prior_estimated_local_min;
    
    for (int step = 0; step < num_steps; ++step) {
        Vector3d pos_c = ugv_pos.back();
        vector<Vector3d> obstacles_in_range = sensor.get_obstacles_in_recognized_area(pos_c, obstacles);
        
        Vector3d F_att = att_force(pos_c, goal_pos);
        Vector3d F_rep = rep_force_optimized(pos_c, obstacles_in_range);
        Vector3d F_resultant = F_att + F_rep;
        
        if (sensor.check_for_parallel_forces(pos_c, F_att, F_rep) && F_rep.norm() > 1e-8) {
            estimated_local_min = find_potential_local_min(pos_c, goal_pos, obstacles_in_range, prior_estimated_local_min, 1000, 1e-8);
            prior_estimated_local_min = estimated_local_min;
        }

        Vector3d vel = F_resultant * STEP_SIZE;
        if (vel.norm() > MAX_VEL) {
            vel = vel.normalized() * MAX_VEL;
        }
        Vector3d pos_n = pos_c + vel;
        ugv_pos.push_back(pos_n);
        
        if ((pos_n - goal_pos).norm() < 0.75 || is_local_min(F_resultant)) {
            break;
        }
    }
    return ugv_pos;
}

// Run a single simulation
void run_single_simulation(int sim_num, double grid_size, int run_index, const Vector3d& predef_ugv_pos, const Vector3d& predef_goal_pos, const string& obstacle_csv_path) {
    string log_dir = "./logs/" + to_string(run_index) + "-(" + to_string(grid_size) + "x" + to_string(grid_size) + ")/" + to_string(sim_num) + "/";
    filesystem::create_directories(log_dir);

    Vector3d pos_i = predef_ugv_pos;
    Vector3d goal_pos = predef_goal_pos;

    vector<Vector3d> obstacles;
    if (!obstacle_csv_path.empty()) {
        ifstream file(obstacle_csv_path);
        string line;
        while (getline(file, line)) {
            istringstream ss(line);
            double x, y;
            char comma;
            ss >> x >> comma >> y;
            obstacles.push_back(Vector3d(x, y, 0));
        }
    } else {
        obstacles = generate_obstacles(12, 0.05, {5, 10}, {goal_pos[0] - 2.5, goal_pos[1] - 2.5, pos_i[0] + 2.5, pos_i[1] + 2.5});
    }

    ofstream obstacle_file(log_dir + "obstacles.csv");
    for (const auto& obs : obstacles) {
        obstacle_file << obs[0] << "," << obs[1] << endl;
    }

    ofstream ugv_goal_file(log_dir + "ugv_goal.csv");
    ugv_goal_file << "ugv_start_x,ugv_start_y,ugv_start_z,goal_x,goal_y,goal_z,grid_size" << endl;
    ugv_goal_file << pos_i[0] << "," << pos_i[1] << "," << pos_i[2] << "," << goal_pos[0] << "," << goal_pos[1] << "," << goal_pos[2] << "," << grid_size << endl;

    vector<Vector3d> ugv_pos = sim_movement_with_DBF(pos_i, goal_pos, obstacles, 10000);

    ofstream log_file(log_dir + "Log.txt");
    for (const auto& pos : ugv_pos) {
        log_file << pos[0] << "," << pos[1] << "," << pos[2] << endl;
    }
}

// Run multiple simulations
void run_multiple_simulations(int n, double grid_size, const Vector3d& predef_ugv_pos, const Vector3d& predef_goal_pos, const string& obstacle_csv_path) {
    vector<future<void>> futures;
    int run_index = 0;

    for (int i = 0; i < n; ++i) {
        futures.push_back(async(run_single_simulation, i, grid_size, run_index, predef_ugv_pos, predef_goal_pos, obstacle_csv_path));
    }

    for (auto& f : futures) {
        f.get();
    }

    cout << "All simulations completed." << endl;
}

int main() {
    Vector3d predef_ugv_pos(2, 3, 0);
    Vector3d predef_goal_pos(8, 5, 0);
    run_multiple_simulations(15, 100, predef_ugv_pos, predef_goal_pos, "");
    return 0;
}