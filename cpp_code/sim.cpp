#include "Sensor.hpp"
#include "DynamicBayesianFiltering.hpp"
#include "graphs.hpp"
//g++ -O3 sim.cpp DynamicBayesianFiltering.cpp Sensor.cpp graphs.hpp -o sim.exe
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <mutex>

using namespace std;
using namespace plotting; 

// Constants for Potential Field Algorithm
const double K_ATT = 1.0;     // Attractive constant
const double K_REP = 0.5;     // Repulsive constant
const double D_SAFE = 5.0;    // Safe distance / Scanner Radius

// Constants for testing simulation
const double STEP_SIZE = 0.01;
const double OBSTACLE_HEIGHT = 10.0; // Max height of obstacles in field graph
const double MAX_VEL = 0.1;          // Max velocity of UGV

// Returns attractive force
// F_att = K_ATT * (goal_pos - ugv_pos)
array<double, 2> att_force(const array<double, 2>& ugv_pos, const array<double, 2>& goal_pos) {
    return {K_ATT * (goal_pos[0] - ugv_pos[0]), K_ATT * (goal_pos[1] - ugv_pos[1])};
}

// Returns repulsive force
array<double, 2> rep_force_optimized(const array<double, 2>& ugv_pos, const vector<array<double, 2>>& obstacles) {
    array<double, 2> total_force = {0.0, 0.0};
    const double epsilon = 1e-25;

    for (const auto& obstacle : obstacles) {
        array<double, 2> displacement_vector = {obstacle[0] - ugv_pos[0], obstacle[1] - ugv_pos[1]};
        double distance = sqrt(displacement_vector[0] * displacement_vector[0] + displacement_vector[1] * displacement_vector[1]);
        // Ignore obstacles beyond D_SAFE
        if (distance >= D_SAFE || distance == 0.0) continue;

        array<double, 2> direction = {-displacement_vector[0] / (distance + epsilon), -displacement_vector[1] / (distance + epsilon)};
        double factor = K_REP * ((1 / (distance + epsilon) - 1 / D_SAFE) * (1 / pow(distance + epsilon, 2)));
        total_force[0] += factor * direction[0];
        total_force[1] += factor * direction[1];
    }

    return total_force;
}

// Checks if the UGV is at a local minimum
bool is_local_min(const array<double, 2>& force_net, double threshold = 1e-3) {
    double force_mag = sqrt(force_net[0] * force_net[0] + force_net[1] * force_net[1]);
    return force_mag < threshold;
}

// Helper function to calculate the net force magnitude at a point
double calc_net_force_mag(const array<double, 2>& pos, const array<double, 2>& goal_pos, const vector<array<double, 2>>& obstacles) {
    array<double, 2> F_att = att_force(pos, goal_pos);
    array<double, 2> F_rep = rep_force_optimized(pos, obstacles);
    double net_force_x = F_att[0] + F_rep[0];
    double net_force_y = F_att[1] + F_rep[1];
    return sqrt(net_force_x * net_force_x + net_force_y * net_force_y);
}

// Struct to hold results of local minimum search
struct LocalMinResult {
    array<double, 2> local_min;
    array<double, 2> F_att;
    array<double, 2> F_rep; 
    array<double, 2> F_net;
    double force_mag;
    bool success;
    int iterations;
};

// Finds the closest potential local minimum within D_SAFE radius
LocalMinResult find_potential_local_min(const array<double, 2>& ugv_pos, const array<double, 2>& goal_pos,
                                      const vector<array<double, 2>>& obstacles, const array<double, 2>& guess_i = {0.0, 0.0},
                                      int max_iter = 1000, double threshold = 1e-3) {
    // Grid search within D_SAFE radius to find good starting point
    array<double, 2> best_pos = ugv_pos;
    // array<double, 2> temp = {0.0, 0.0};
    // array<double, 2> best_pos = (guess_i == temp) ? ugv_pos : guess_i;
    double min_force = calc_net_force_mag(ugv_pos, goal_pos, obstacles);
    
    // Search in a grid pattern within D_SAFE radius
    // Calculate direction vector from UGV to goal
    double goal_dir_x = goal_pos[0] - ugv_pos[0];
    double goal_dir_y = goal_pos[1] - ugv_pos[1];
    double goal_angle = atan2(goal_dir_y, goal_dir_x);

    double search_step = D_SAFE / 100.0;
    for(double r = 0; r <= D_SAFE; r += search_step) {
        // Search in an arc centered on goal direction
        for(double theta = goal_angle - M_PI/2; theta <= goal_angle + M_PI/2; theta += search_step/r) {
            // Convert polar to cartesian coordinates
            double x = ugv_pos[0] + r * cos(theta);
            double y = ugv_pos[1] + r * sin(theta);
            
            array<double, 2> test_pos = {x, y};
            double force = calc_net_force_mag(test_pos, goal_pos, obstacles);
            if(force < min_force) {
                min_force = force;
                best_pos = test_pos;
            }
        }
    }

    // Gradient descent from best grid point
    array<double, 2> current_pos = best_pos;
    double current_force = min_force;
    int iter = 0;
    
    while(current_force > threshold && iter < max_iter) {
        array<double, 2> grad = {0.0, 0.0};
        // Calculate gradient
        for(int i = 0; i < 2; i++) {
            array<double, 2> perturbed_pos = current_pos;
            perturbed_pos[i] += 0.0001;
            double perturbed_force = calc_net_force_mag(perturbed_pos, goal_pos, obstacles);
            grad[i] = (perturbed_force - current_force) / 0.0001;
        }
        
        // Update position (staying within D_SAFE radius)
        array<double, 2> new_pos = {
            current_pos[0] - grad[0] * STEP_SIZE,
            current_pos[1] - grad[1] * STEP_SIZE
        };
        
        // Ensure new position is within D_SAFE radius
        double dist_to_ugv = sqrt(pow(new_pos[0] - ugv_pos[0], 2) + pow(new_pos[1] - ugv_pos[1], 2));
        if(dist_to_ugv > D_SAFE) {
            // Project back onto D_SAFE circle
            double angle = atan2(new_pos[1] - ugv_pos[1], new_pos[0] - ugv_pos[0]);
            new_pos[0] = ugv_pos[0] + D_SAFE * cos(angle);
            new_pos[1] = ugv_pos[1] + D_SAFE * sin(angle);
        }
        
        double new_force = calc_net_force_mag(new_pos, goal_pos, obstacles);
        if(new_force >= current_force) break; // Stop if force increases
        
        current_pos = new_pos;
        current_force = new_force;
        iter++;
    }
    // array<double, 2> current_pos = best_pos;
    // double current_force = min_force;
    // int iter = 0;

    // Compute final forces
    array<double, 2> final_F_att = att_force(current_pos, goal_pos);
    array<double, 2> final_F_rep = rep_force_optimized(current_pos, obstacles);
    array<double, 2> final_F_net = {final_F_att[0] + final_F_rep[0], final_F_att[1] + final_F_rep[1]};
    
    return {current_pos, final_F_att, final_F_rep, final_F_net, current_force, true, iter};
}

// Generates clustered obstacles
vector<array<double, 2>> generate_obstacles(int num_obstacles, double step_size = 0.001, pair<double, double> length_range = {5.0, 20.0},
                                            array<double, 4> area_size = {100.0, 100.0, 0.0, 0.0}) {
    vector<array<double, 2>> obstacles;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis_area_x(area_size[2], area_size[0]);
    uniform_real_distribution<> dis_area_y(area_size[3], area_size[1]);
    uniform_real_distribution<> dis_angle(0.0, 2 * M_PI);
    uniform_real_distribution<> dis_length(length_range.first, length_range.second);

    // Direction vector and perpendicular vector for tunnel
    array<double, 2> direction_vector = {area_size[0] - area_size[2], area_size[1] - area_size[3]};
    double norm = sqrt(direction_vector[0] * direction_vector[0] + direction_vector[1] * direction_vector[1]);
    direction_vector[0] /= norm;
    direction_vector[1] /= norm;
    array<double, 2> perpendicular_vector = {-direction_vector[1], direction_vector[0]};

    double tunnel_width = 12.5;

    for (int i = 0; i < num_obstacles; ++i) {
        uniform_real_distribution<> dis_distance(0.0, norm);
        double distance_along_tunnel = dis_distance(gen);
        double start_x = area_size[2] + distance_along_tunnel * direction_vector[0];
        double start_y = area_size[3] + distance_along_tunnel * direction_vector[1];

        uniform_real_distribution<> dis_width_offset(-tunnel_width / 2, tunnel_width / 2);
        double width_offset = dis_width_offset(gen);
        start_x += width_offset * perpendicular_vector[0];
        start_y += width_offset * perpendicular_vector[1];

        // Ensure obstacle is within boundaries
        start_x = max(min(start_x, area_size[0]), area_size[2]);
        start_y = max(min(start_y, area_size[1]), area_size[3]);

        double angle = dis_angle(gen);
        double length = dis_length(gen);

        int num_points = static_cast<int>(length / step_size) + 1;
        for (int j = 0; j < num_points; ++j) {
            double offset = step_size * j;
            double x = start_x + offset * cos(angle);
            double y = start_y + offset * sin(angle);
            obstacles.push_back({x, y});
        }
    }

    return obstacles;
}

// Simulation of movement using Potential Field Algorithm with DBF
tuple<vector<array<double, 2>>, vector<double>, string, array<double, 2>>
sim_movement_with_DBF(const array<double, 2>& pos_i, const array<double, 2>& goal_pos,
                      const vector<array<double, 2>>& obstacles, int num_steps) {

    vector<array<double, 2>> ugv_pos = {pos_i};
    vector<double> speeds;
    Sensor sensor(D_SAFE, 180, 1000);
    DBF dbf;
    bool first_check = true;
    int previous_num_obs = 0;
    string log_string;
    LocalMinResult estimated_local_min;
    LocalMinResult prior_estimated_local_min;
    array<double, 2> empty_array = {0.0, 0.0};
    array<double, 2> et_local_min = {0.0, 0.0};

    for (int step = 0; step < num_steps; ++step) {
        array<double, 2> pos_c = ugv_pos.back();

        // Convert obstacles to 3D
        vector<array<double, 3>> obstacles_3d;
        for (const auto& obs : obstacles) {
            obstacles_3d.push_back({obs[0], obs[1], 0.0});
        }

        // Get obstacles in range
        vector<array<double, 3>> obstacles_in_range = sensor.get_obstacles_in_recognized_area({pos_c[0], pos_c[1], 0.0}, obstacles_3d);

        // Potential Field Algorithm
        array<double, 2> F_att = att_force(pos_c, goal_pos);
        array<double, 2> F_resultant = F_att;

        if (!obstacles_in_range.empty()) {
            vector<array<double, 2>> obstacles_in_range_2d;
            for (const auto& obs : obstacles_in_range) {
                obstacles_in_range_2d.push_back({obs[0], obs[1]});
            }
            array<double, 2> F_rep = rep_force_optimized(pos_c, obstacles_in_range_2d);
            F_resultant[0] += F_rep[0];
            F_resultant[1] += F_rep[1];
        }

        // Check for parallel forces
        if (sensor.check_for_parallel_forces({F_att[0], F_att[1], 0.0}, {F_resultant[0], F_resultant[1], 0.0})) {
            vector<array<double, 2>> obstacles_in_range_2d;
            for (const auto& obs : obstacles_in_range) {
                obstacles_in_range_2d.push_back({obs[0], obs[1]});
            }
            if (first_check || prior_estimated_local_min.local_min == empty_array) {
                estimated_local_min = find_potential_local_min(pos_c, goal_pos, obstacles_in_range_2d);
            } else {
                estimated_local_min = find_potential_local_min(pos_c, goal_pos, obstacles_in_range_2d, prior_estimated_local_min.local_min);
            }

            if (!first_check && (abs(estimated_local_min.local_min[0] - prior_estimated_local_min.local_min[0]) > 2.00 ||
                                 abs(estimated_local_min.local_min[1] - prior_estimated_local_min.local_min[1]) > 2.00)) {
                first_check = true;
            } else if (!first_check) {
                estimated_local_min = prior_estimated_local_min;
            }
            if (first_check) {
                int group_a_points = sensor.get_group_a_points({pos_c[0], pos_c[1], 0.0}, {estimated_local_min.local_min[0], estimated_local_min.local_min[1], 0.0}, 1.0);
                log_string += "Initial belief: " + to_string(dbf.initialize_belief(group_a_points)) + "\n";
                log_string += "Group A Points: " + to_string(group_a_points) + "\n";
                first_check = false;
                prior_estimated_local_min = estimated_local_min;
            }

            string log_str;
            SensorData sensorData = sensor.get_sensor_data({pos_c[0], pos_c[1], 0.0}, {estimated_local_min.local_min[0], estimated_local_min.local_min[1], 0.0}, obstacles_in_range, log_str);
            auto [belief, reached_threshold, dbf_log] = dbf.update_belief(sensorData);

            if (belief > 0.1) {
                log_string += to_string(step) + "th Step\n";
                log_string += dbf_log;
                log_string += log_str + "; ";
                log_string += to_string(estimated_local_min.local_min[0]) + ", " + to_string(estimated_local_min.local_min[1]) + " Estimated Local Min\n";
                log_string += to_string(estimated_local_min.force_mag) + " Force Magnitude\n";
                log_string += to_string(estimated_local_min.iterations) + " Iterations\n";
                log_string += to_string(estimated_local_min.success) + " Success\n";
                log_string += to_string(obstacles_in_range.size()) + " Obstacles in Range\n";
                log_string += to_string(belief) + " Belief\n";
            } else {
                log_string += to_string(step) + " Step\n";
                log_string += dbf_log;
                log_string += log_str + ";\n";
                log_string += to_string(estimated_local_min.local_min[0]) + ", " + to_string(estimated_local_min.local_min[1]) + " Estimated Local Min\n";
                log_string += to_string(estimated_local_min.force_mag) + " Force Magnitude\n";
                log_string += to_string(estimated_local_min.success) + " Success\n";
                log_string += "UGV_POS: (" + to_string(pos_c[0]) + ", " + to_string(pos_c[1]) + ")\n";
                log_string += " Belief too low, moving on...\n\n";
            }

            if (reached_threshold) {
                log_string += "Local Min Predicted!, Current Step: " + to_string(step) + "\n";
                vector<double> pos_c_vec = {pos_c[0], pos_c[1]};
                vector<double> local_min_vec = {estimated_local_min.local_min[0], estimated_local_min.local_min[1]};
                int steps_away = steps_to_local_min(pos_c_vec, local_min_vec, STEP_SIZE);
                log_string += to_string(steps_away) + " Steps Away\n";
                log_string += to_string(estimated_local_min.local_min[0]) + ", " + to_string(estimated_local_min.local_min[1]) + " Estimated Local Min\n";
                break;
            }
        } else {
            first_check = true;
        }

        // Update speed and position
        double speed = sqrt(F_resultant[0] * F_resultant[0] + F_resultant[1] * F_resultant[1]);
        speeds.push_back(speed);
        array<double, 2> vel = {F_resultant[0] * STEP_SIZE, F_resultant[1] * STEP_SIZE};
        double vel_norm = sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
        if (vel_norm > MAX_VEL) {
            vel[0] = vel[0] / vel_norm * MAX_VEL;
            vel[1] = vel[1] / vel_norm * MAX_VEL;
        }
        array<double, 2> pos_n = {pos_c[0] + vel[0], pos_c[1] + vel[1]};
        ugv_pos.push_back(pos_n);

        // Check if goal is reached
        if (sqrt((pos_n[0] - goal_pos[0]) * (pos_n[0] - goal_pos[0]) + (pos_n[1] - goal_pos[1]) * (pos_n[1] - goal_pos[1])) < 0.75) {
            log_string += "Goal Reached, step: " + to_string(step) + "\n";
            break;
        }

        // Check if local minimum is found
        if (is_local_min(F_resultant)) {
            log_string += "Local Min Found, step: " + to_string(step) + "\n";
            log_string += "UGV Position: (" + to_string(pos_n[0]) + ", " + to_string(pos_n[1]) + ")\n";
            log_string += "Wasn't predicted :( (bad)\n";
            break;
        }
    }

    return make_tuple(ugv_pos, speeds, log_string, estimated_local_min.local_min);
}

// Helper function to generate a random double between min and max
double getRandomDouble(double min, double max) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Runs a single simulation
void run_single_simulation(int sim_num, int grid_size, int run_index,
                           const array<double, 2>& predef_ugv_pos = {0.0, 0.0},
                           const array<double, 2>& predef_goal_pos = {0.0, 0.0},
                           const string& obstacle_csv_path = "") {

    string log_dir = "./logs/";
    string sim_id = "sim" + to_string(sim_num) + "_" + to_string(run_index);
    array<double, 2> pos_i = predef_ugv_pos != array<double, 2>{0.0, 0.0} ? predef_ugv_pos :
        array<double, 2>{getRandomDouble(0.0, grid_size), getRandomDouble(0.0, grid_size)};
    array<double, 2> goal_pos = predef_goal_pos != array<double, 2>{0.0, 0.0} ? predef_goal_pos :
        array<double, 2>{getRandomDouble(0.0, grid_size), getRandomDouble(0.0, grid_size)};

    vector<array<double, 2>> obstacles;
    if (!obstacle_csv_path.empty()) {
        ifstream infile(obstacle_csv_path);
        string line;
        while (getline(infile, line)) {
            stringstream ss(line);
            string x_str, y_str;
            getline(ss, x_str, ',');
            getline(ss, y_str, ',');
            obstacles.push_back({stod(x_str), stod(y_str)});
        }
    } else {
        int num_obstacles = rand() % (grid_size * grid_size / 50 - grid_size * grid_size / 100 + 1) + grid_size * grid_size / 100;
        obstacles = generate_obstacles(12, 0.05, {5.0, 10.0}, {goal_pos[0] - 2.5, goal_pos[1] - 2.5, pos_i[0] + 2.5, pos_i[1] + 2.5});
    }

    string sim_dir = "./logs/" + to_string(run_index) + "-(" + to_string(grid_size) + "x" + to_string(grid_size) + ")/" + to_string(sim_num) + "/";
    filesystem::create_directories(sim_dir);
    string log_filename = sim_dir + "Log_" + sim_id + ".txt";
    string obstacle_csv_filename = sim_dir + "obstacles_" + sim_id + ".csv";
    string ugv_goal_csv_filename = sim_dir + "ugv_goal_" + sim_id + ".csv";

    // Save obstacle positions to CSV
    ofstream obstacle_file(obstacle_csv_filename);
    for (const auto& obs : obstacles) {
        obstacle_file << obs[0] << "," << obs[1] << "\n";
    }
    obstacle_file.close();

    // Save UGV start and goal positions to CSV
    ofstream ugv_goal_file(ugv_goal_csv_filename);
    ugv_goal_file << "ugv_start_x,ugv_start_y,ugv_start_z,goal_x,goal_y,goal_z,grid_size\n";
    ugv_goal_file << pos_i[0] << "," << pos_i[1] << ",0.0," << goal_pos[0] << "," << goal_pos[1] << ",0.0," << grid_size << "\n";
    ugv_goal_file.close();

    cout << "\nSimulation " << sim_num << " started.\n";

    // Run the simulation
    auto [ugv_pos, speeds, log_string, et_local_min] = sim_movement_with_DBF(pos_i, goal_pos, obstacles, 10000);

    // Save log
    ofstream log_file(log_filename);
    log_file << log_string;
    log_file.close();

    Point start = {pos_i[0], pos_i[1]};
    Point goal = {goal_pos[0], goal_pos[1]};
    Point local_min = {et_local_min[0], et_local_min[1]};
    std::vector<Point> obs;
    for (const auto& obs_pos : obstacles) {
        obs.push_back({obs_pos[0], obs_pos[1]});
    }
    std::vector<Point> ugvPos;
    for (const auto& pos : ugv_pos) {
        ugvPos.push_back({pos[0], pos[1]});
    }
    std::vector<double> obsContainer = {goal_pos[0] - 2.5, goal_pos[1] - 2.5, pos_i[0] + 2.5, pos_i[1] + 2.5};
    SVGPlotter::plotMovementInteractive2D(ugvPos, goal, obs, sim_dir, sim_id, obsContainer, local_min);     

    cout << "\nSimulation " << sim_num << " completed.\n";
}

// Runs multiple simulations
void run_multiple_simulations(int n, int grid_size,
                              const array<double, 2>& predef_ugv_pos = {0.0, 0.0},
                              const array<double, 2>& predef_goal_pos = {0.0, 0.0},
                              const string& obstacle_csv_path = "") {

    auto time_start = chrono::high_resolution_clock::now();
    string log_dir = "./logs/";
    filesystem::create_directories(log_dir);

    vector<int> file_nums;
    for (const auto& entry : filesystem::directory_iterator(log_dir)) {
        string filename = entry.path().filename().string();
        size_t pos = filename.find('-');
        if (pos != string::npos && all_of(filename.begin(), filename.begin() + pos, ::isdigit)) {
            file_nums.push_back(stoi(filename.substr(0, pos)));
        }
    }
    int run_index = file_nums.empty() ? 0 : *max_element(file_nums.begin(), file_nums.end()) + 1;

    vector<future<void>> futures;
    for (int sim_num = 0; sim_num < n; ++sim_num) {
        futures.push_back(async(launch::async, run_single_simulation, sim_num, grid_size, run_index,
                                predef_ugv_pos, predef_goal_pos, obstacle_csv_path));
    }

    for (auto& f : futures) {
        f.get();
    }

    string sim_dir = "./logs/" + to_string(run_index) + "-(" + to_string(grid_size) + "x" + to_string(grid_size) + ")/";
    cout << "\n\nAll " << n << " simulations completed. Results saved to " << sim_dir << endl;
    auto time_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(time_end - time_start).count();
    int minutes = duration / 60;
    int seconds = duration % 60;
    cout << "Time taken: " << minutes << " min, " << seconds << " seconds\n";
}

int main() {
    // run a sim with 15 simulations, 100x100 grid, random UGV start and goal positions, and random obstacles:
    run_multiple_simulations(15, 100, {2.0, getRandomDouble(5.0, 30.0)}, {92.0, getRandomDouble(50.0, 98.0)});
    return 0;
}
