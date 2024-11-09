#include <matplot/matplot.h>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace matplot;

void plot_movement_interactive_2d(const vector<array<double, 2>>& ugv_pos, 
                                  const array<double, 2>& goal_pos, 
                                  const vector<array<double, 2>>& obstacles, 
                                  const string& log_dir, 
                                  const string& sim_id, 
                                  const vector<double>& obs_container, 
                                  const array<double, 2>& et_local_min) {
    // Create the plot
    auto fig = gcf();
    auto ax = gca();

    // Add the UGV path
    vector<double> ugv_x, ugv_y;
    for (const auto& pos : ugv_pos) {
        ugv_x.push_back(pos[0]);
        ugv_y.push_back(pos[1]);
    }
    ax->scatter(ugv_x, ugv_y, 5.0)->line_width(2).color("blue").display_name("UGV Path");

    // Add the goal position
    ax->scatter({goal_pos[0]}, {goal_pos[1]}, 10.0)->marker_face(true).color("black").display_name("Goal");

    // Add the starting position
    ax->scatter({ugv_pos[0][0]}, {ugv_pos[0][1]}, 10.0)->marker_face(true).color("green").display_name("Start");

    // Add the local minimum
    ax->scatter({et_local_min[0]}, {et_local_min[1]}, 10.0)->marker_face(true).color("black").display_name("Local Minimum");

    // Add obstacles
    vector<double> obs_x, obs_y;
    for (const auto& obs : obstacles) {
        obs_x.push_back(obs[0]);
        obs_y.push_back(obs[1]);
    }
    ax->scatter(obs_x, obs_y, 5.0)->marker_face(true).color("red").display_name("Obstacles");

    // Plot scanner trace
    for (const auto& pos : ugv_pos) {
        vector<double> x_scan, y_scan;
        for (double theta = -M_PI/2; theta <= M_PI/2; theta += M_PI/50) {
            x_scan.push_back(pos[0] + 5 * cos(theta));
            y_scan.push_back(pos[1] + 5 * sin(theta));
        }
        ax->plot(x_scan, y_scan)->line_width(1).line_style("--").color("orange").display_name("Scanner Trace").opacity(0.5);
    }

    // Draw a dot for each point in obs_container and shade the inside red
    vector<double> obs_container_x = {obs_container[0], obs_container[2], obs_container[4], obs_container[6]};
    vector<double> obs_container_y = {obs_container[1], obs_container[3], obs_container[5], obs_container[7]};
    ax->scatter(obs_container_x, obs_container_y, 5.0)->marker_face(true).color("purple").display_name("Obstacle Container Points");
    obs_container_x.push_back(obs_container_x[0]);
    obs_container_y.push_back(obs_container_y[0]);
    ax->fill(obs_container_x, obs_container_y, "purple")->face_alpha(0.2).display_name("Obstacle Container");

    // Set plot layout for better readability
    ax->title("UGV Path with Scanner Trace + Container");
    ax->xlabel("X");
    ax->ylabel("Y");
    ax->legend();
    ax->size(800, 800);

    // Save the plot as an interactive HTML file
    ofstream html_file(log_dir + "/movement_" + sim_id + ".html");
    html_file << fig->render();
    html_file.close();
}