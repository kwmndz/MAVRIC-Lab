import numpy as np
import Graphs as g
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Sensor import Sensor
from DynamicBayesianFiltering import DBF, SensorData, steps_to_local_min
import sys
import os

# Constants for Potential Field Algorithm
K_ATT = 1.0 # Attractive
K_REP = 0.5 # Repulsive
D_SAFE = 2 # Safe distance / Scanner Radius

# Constants for testing simulation (keep low or potential field algorithm gives weird results)
STEP_SIZE = 0.01
OBSTACLE_HEIGHT = 10 # max height of obstacles in field graph


"""

    Potential Field Algorithm Calculations
    https://www.youtube.com/watch?v=Ls8EBoG_SEQ
    Works for 3D and 2D
     
"""


# Returns attractive force
# Assuming goal_pos and ugv_pos are vectors
# F_att =  K_att * (goal_pos - ugv_pos)
def att_force(ugv_pos, goal_pos):
    return K_ATT * (np.array(goal_pos) - np.array(ugv_pos))


# Returns repulsive force
# Assuming obs_pos and ugv_pos are vectors
# F_rep = K_rep * (1/distance - 1/d_safe) * (1/distance^3) * (obs_pos - ugv_pos)
def rep_force(ugv_pos, obs_pos):
    # maginitude of distance
    distance = np.linalg.norm(np.array(obs_pos) - np.array(ugv_pos))
    
    if distance >= D_SAFE: # not close enough to obstacle
        if len(ugv_pos) == 3:
            return np.array([0.0, 0.0, 0.0])
        
        return np.array([0.0, 0.0])
    
    # unit vector of direction
    direction = (np.array(ugv_pos) - np.array(obs_pos)) / distance
    return K_REP * (1/distance - 1/D_SAFE) * (1/(distance**2)) * direction

# Checks if the ugv is at a local min
def is_local_min(force_net, threshold=1e-3):
    force_mag = np.linalg.norm(force_net)
    
    # If force is below small threshold, then it is a local min
    return force_mag < threshold

# helper method to calculate the force difference 
def force_difference(ugv_pos, goal_pos, obstacles):
    F_att = att_force(ugv_pos, goal_pos)    
    
    F_rep = np.zeros(len(ugv_pos))
    for obs in obstacles:
        F_rep += rep_force(ugv_pos, obs)
    
    return np.linalg.norm(F_att + F_rep)

# Finds the potential local min using the Potential Field Algorithm and the Simplex Method (Nelder-Mead)
# https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method <-- More info on Nelder-Mead method
# Returns a dictionary with a bunch of info mostly for testing/debugging purposes, all thats really needed is the 'local_min'
def find_potential_local_min(ugv_pos, goal_pos, obstacles, guess_i = None, max_iter = 1000, threshold = 1e-8):
    
    if guess_i is None:
        midpoint_obstacles = np.mean(obstacles, axis=0)
        guess_i = (ugv_pos + midpoint_obstacles) / 2
    
    # Confusing stuff, dont worry about it, all you need to know is that it minimizes the value of the 
    # force difference where the only non-constant is x; where x is the ugv_pos  
    objective_func = lambda x: force_difference(x, goal_pos, obstacles)
    result = minimize(objective_func, guess_i, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': threshold})
    
    # Get forces to check that the minizmation worked
    F_att = att_force(result.x, goal_pos)
    F_rep = np.zeros(len(result.x))
    for obs in obstacles:
        F_rep += rep_force(result.x, obs)

    
    return {
        'local_min': result.x,
        'F_att': F_att,
        'F_rep': F_rep,
        'F_net': F_att + F_rep,
        'force_mag': result.fun,
        'success': result.success,
        'iterations': result.nit
    }


"""

    Simulation of movement using Potential Field Algorithm
    For testing purposes
    
"""
# Simulates the movement of the UGV in 2D or 3D
def sim_movement(pos_i, goal_pos, obstacles, num_steps):
    ugv_pos = [pos_i]
    speeds = []
    
    for _ in range(num_steps):
        pos_c = ugv_pos[-1] # Always get last added position
        
        # Potential Field Algorithm for force calcs
        F_att = att_force(pos_c, goal_pos)
        
        if len(pos_c) == 3: # 3D
            F_rep = np.array([0.0, 0.0, 0.0])
        else: # 2D
            F_rep = np.array([0.0, 0.0])
            
        for obs in obstacles:
            F_rep += rep_force(pos_c, obs)
        
        F_resultant = F_att + F_rep
        
        speed = np.linalg.norm(F_resultant)  
        speeds.append(speed)
        
        # Update position
        pos_n = pos_c + F_resultant * STEP_SIZE
        ugv_pos.append(pos_n)
        
        if is_local_min(F_resultant):
            print('Local Min Found')
            break # ends sim
        
    return np.array(ugv_pos), np.array(speeds)

def sim_movement_with_DBF(pos_i, goal_pos, obstacles, num_steps):
    ugv_pos = [pos_i]
    speeds = []
    sensor = Sensor(D_SAFE, 180, 10000)
    dbf = DBF()
    first_check = True
    previus_num_obs = 0
    
    for _ in range(num_steps):
        pos_c = ugv_pos[-1] # Always get last added position
        
        obstacles_in_range = sensor.get_obstacles_in_recognized_area(pos_c, obstacles)
        
        # Potential Field Algorithm for force calcs
        F_att = att_force(pos_c, goal_pos)
        
        F_rep = np.zeros(len(pos_c))
            
        for obs in obstacles_in_range:
            F_rep += rep_force(pos_c, obs)
        
        F_resultant = F_att + F_rep
        
        if len(obstacles_in_range) != previus_num_obs:
            first_check = True
            previus_num_obs = len(obstacles_in_range)
        
        if sensor.check_for_parrallel_forces(pos_c, F_att, F_rep) and np.linalg.norm(F_rep) > 1e-8:
            if first_check:
                estimated_local_min = find_potential_local_min(pos_c, goal_pos, obstacles_in_range)
                group_a_points = sensor.get_group_a_points(pos_c, estimated_local_min['local_min'], step_size = 0.1)
                print(dbf.initialize_belief(group_a_points))
                first_check = False
                
            sensorData = sensor.get_sensor_data(pos_c, estimated_local_min['local_min'], obstacles_in_range)
            print(sensorData)
            #print(pos_c)
            #print(obstacles_in_range)
            belief, reached_threshold = dbf.update_belief(sensorData)
            
            if belief > 0.1:
                print(_, "Step")
                print(estimated_local_min['local_min'], "Estimated Local Min")
                print(estimated_local_min['force_mag'], "Force Magnitude")
                print(estimated_local_min['iterations'], "Iterations")
                print(estimated_local_min['success'], "Success")
                print(estimated_local_min['F_att'], "F_att")
                print(estimated_local_min['F_rep'], "F_rep")
                print(obstacles_in_range, "Obstacles in Range")
                print(belief, "Belief")
                print("\n\n")
            
            if reached_threshold:
                print('Local Min Found')
                steps_away = steps_to_local_min(pos_c, estimated_local_min['local_min'], step_size=STEP_SIZE)
                print(steps_away, "Steps Away")
                print(estimated_local_min['local_min'], "Estimated Local Min")
                print("\n\n")
                break
        
        else:
            first_check = True
            
                
        speed = np.linalg.norm(F_resultant)  
        speeds.append(speed)
        
        # Update position
        pos_n = pos_c + F_resultant * STEP_SIZE
        ugv_pos.append(pos_n)
        
        if np.linalg.norm(pos_n - goal_pos) < 0.75:
            print('Goal Reached')
            break
        
        if is_local_min(F_resultant):
            print('Local Min Found')
            print(pos_n)
            break
        
    return np.array(ugv_pos), np.array(speeds)


# Testing the simulation
if __name__ == '__main__':
    # testing constants
    pos_i = np.array([2, 3, 0])
    goal_pos = np.array([8, 5, 0])
    obstacles = [[7, y, 0] for y in range(3, 8)]
    
    
    midpoint = (pos_i + goal_pos) / 2.0 # midpoint between start and goal
    obstacles.append([midpoint[0], midpoint[1], midpoint[2]])
    
    
    obstacles = np.array(obstacles)
    
    # amt of steps to simulate
    num_steps = 10000
    
    # constants for potential field surface graph
    field_size = 10
    res = 100 # higher res = more detail but more lag
    
    # Redirect print out to log file
    log_dir = './logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('Log_') and f.endswith(f'_{STEP_SIZE}.txt')]
    if log_files:
        max_index = max([int(f.split('_')[1]) for f in log_files])
        log_filename = f'Log_{max_index + 1}_{STEP_SIZE}.txt'
    else:
        log_filename = f'Log_0_{STEP_SIZE}.txt'

    sys.stdout = open(os.path.join(log_dir, log_filename), 'w')
    
    ugv_pos, speeds = sim_movement_with_DBF(pos_i, goal_pos, obstacles, num_steps)
    
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print('Simulation Done')
    print('Output Logs written to:', log_filename)
    
    """
    Implementation of plotting functions in Graphs.py
    """
    
    #g.plot_speed_time_2d(speeds, len(speeds))
    g.plot_movement_2d(ugv_pos, goal_pos, obstacles)
    g.plot_potential_field_surface(goal_pos, obstacles, field_size, res)
    
    # Allows for all plots to be shown at the same time
    plt.pause(0.001)
    plt.show(block = True)