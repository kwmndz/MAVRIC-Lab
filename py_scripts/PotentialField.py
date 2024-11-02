import numpy as np
import Graphs as g
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Sensor import Sensor
from DynamicBayesianFiltering import DBF, SensorData, steps_to_local_min
import sys
import os
import csv
import concurrent.futures
import time

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
    sensor = Sensor(D_SAFE, 180, 1000)
    dbf = DBF()
    first_check = True
    previus_num_obs = 0
    log_string = ""
    
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
                #print(dbf.initialize_belief(group_a_points))
                log_string += f"Initial belief: {dbf.initialize_belief(group_a_points)}\n"
                first_check = False
                
            sensorData = sensor.get_sensor_data(pos_c, estimated_local_min['local_min'], obstacles_in_range)
            #print(sensorData)
            #print(pos_c)
            #print(obstacles_in_range)
            belief, reached_threshold, log_str = dbf.update_belief(sensorData)
            
            log_entries = []
            if belief > 0.1:
                log_entries.append(f"{_} Step")
                log_entries.append(log_str)
                log_entries.append(f"{estimated_local_min['local_min']} Estimated Local Min")
                log_entries.append(f"{estimated_local_min['force_mag']} Force Magnitude")
                log_entries.append(f"{estimated_local_min['iterations']} Iterations")
                log_entries.append(f"{estimated_local_min['success']} Success")
                log_entries.append(f"{estimated_local_min['F_att']} F_att")
                log_entries.append(f"{estimated_local_min['F_rep']} F_rep")
                log_entries.append(f"{obstacles_in_range} Obstacles in Range")
                log_entries.append(f"{belief} Belief\n")
            else:
                log_entries.append(f"{_} Step")
                log_entries.append(log_str)
                log_entries.append(f"belief too low moving on....\n")
            
            if reached_threshold:
                log_entries.append(f'Local Min Predicted!, Current Step: {_}')
                steps_away = steps_to_local_min(pos_c, estimated_local_min['local_min'], step_size=STEP_SIZE)
                log_entries.append(f"{steps_away} Steps Away")
                log_entries.append(f"{estimated_local_min['local_min']} Estimated Local Min\n")
                log_string += "\n".join(log_entries) + "\n"
                break
            
            log_string += "\n".join(log_entries) + "\n"
        
        else:
            first_check = True
            
                
        speed = np.linalg.norm(F_resultant)  
        speeds.append(speed)
        
        # Update position
        pos_n = pos_c + F_resultant * STEP_SIZE
        ugv_pos.append(pos_n)
        
        if np.linalg.norm(pos_n - goal_pos) < 0.75:
            log_string += f'Goal Reached, step: {_}\n'
            break
        
        if is_local_min(F_resultant):
            log_string += f'Local Min Found, step: {_}\n'
            log_string += f"{pos_n} UGV Position\n"
            log_string += f"Wasnt predicted :( (bad) \n)"
            break
        
    return np.array(ugv_pos), np.array(speeds), log_string

# Helper function to run a single simulation so I can multi-thread it
def run_single_simulation(sim_num, grid_size, run_index, predef_ugv_pos=None, predef_goal_pos=None, obstacle_csv_path=None):
    log_dir = './logs/'
    sim_id = f'sim{sim_num}_{run_index}'
    
    # Set or randomize UGV start and goal positions
    pos_i = np.array(predef_ugv_pos) if predef_ugv_pos is not None else np.array(
        [np.random.uniform(0, grid_size), np.random.uniform(0, grid_size), 0]
    )
    goal_pos = np.array(predef_goal_pos) if predef_goal_pos is not None else np.array(
        [np.random.uniform(0, grid_size), np.random.uniform(0, grid_size), 0]
    )
    
     # Set or randomize obstacle positions
    if obstacle_csv_path:
        with open(obstacle_csv_path, 'r') as f:
            reader = csv.reader(f)
            obstacles = [np.array([float(x), float(y), 0]) for x, y in reader]
    else:
        num_obstacles = np.random.randint(5, 15)
        obstacles = [np.array([np.random.uniform(0, grid_size), np.random.uniform(0, grid_size), 0]) for _ in range(num_obstacles)]

    
    log_dir = f'./logs/{run_index}-({grid_size}x{grid_size})/{sim_num}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = f'{log_dir}Log_{sim_id}.txt'
    obstacle_csv_filename = f'{log_dir}obstacles_{sim_id}.csv'
    ugv_goal_csv_filename = f'{log_dir}ugv_goal_{sim_id}.csv'
    time_taken_filename = f'{log_dir}time_taken_{sim_id}.txt'
    
    # Save obstacle positions to CSV
    with open(obstacle_csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([[obs[0], obs[1]] for obs in obstacles])
    
    # Save UGV start and goal positions to CSV
    with open(ugv_goal_csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ugv_start_x', 'ugv_start_y', 'ugv_start_z', 'goal_x', 'goal_y', 'goal_z', 'grid_size'])
        writer.writerow([pos_i[0], pos_i[1], pos_i[2], goal_pos[0], goal_pos[1], goal_pos[2], grid_size])
    
    #time_start = time.time()
    print(f'\nSimulation {sim_num} started.\n')
    
    # Redirect output to log file
    #sys.stdout = open(log_filename, 'w')
    
    # Run the simulation
    ugv_pos, speeds, log_string = sim_movement_with_DBF(pos_i, goal_pos, obstacles, num_steps=10000)
    
    # Close log file and reset stdout
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    
    # Plot and save plots to log directory
    g.plot_movement_2d(ugv_pos, goal_pos, obstacles, log_dir, sim_id)
    #plt.savefig(f'{log_dir}movement_{sim_id}.png')
    #plt.close()  # Close the plot after saving to avoid displaying
    
    #g.plot_potential_field_surface(goal_pos, obstacles, field_size=grid_size, res=100, log_dir=log_dir, sim_id=sim_id)
    #plt.savefig(f'{log_dir}potential_field_{sim_id}.png')
    #plt.close()
    
    log_dir = f'./logs/{run_index}-({grid_size}x{grid_size})/{sim_num}/'
    print(f'\nSimulation {sim_num} completed.\n')
    #print(f'Time taken: {time.time() - time_start} seconds')
    #with open(time_taken_filename, 'w') as f:
    #    f.write(f'Time taken for simulation: {time.time() - time_start} seconds')
        
    with open(log_filename, 'w') as f:
        f.write(log_string)

# Run multiple simulations with randomized starting, goal, and obstacle positions
# optional parameters to set predefined UGV start and goal positions, and obstacle positions
# obstacle_csv_path is a path to a CSV file containing obstacle positions
# Threading to increase performance 
def run_multiple_simulations(n, grid_size, predef_ugv_pos=None, predef_goal_pos=None, obstacle_csv_path=None):
    time_start = time.time()
    log_dir = './logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # To track number of runs across all simulations
    #run_index = len([f for f in os.listdir(log_dir) if f.startswith('Log_')])
    file_nums = [int(f.split('-')[0]) for f in os.listdir(log_dir) if f.split('-')[0].isdigit()]
    if file_nums:
        run_index = max(file_nums) + 1
    else:
        run_index = 0

    simulation_args = [
        (sim_num, grid_size, run_index, predef_ugv_pos, predef_goal_pos, obstacle_csv_path) 
        for sim_num in range(n)
    ]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_single_simulation, *args) for args in simulation_args]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f'\n\nError in simulation: {e}\n\n')
        
    log_dir = f'./logs/{run_index}-({grid_size}x{grid_size})/'
    print(f'\n\nAll {n} simulations completed. Results saved to {log_dir}')
    time_taken = time.time() - time_start
    minutes = int(time_taken // 60)
    seconds = round(time_taken % 60, 2)
    print(f'Time taken: {minutes} min, {seconds} seconds')

# Testing the simulation
if __name__ == '__main__':
    run_multiple_simulations(5, 10, predef_ugv_pos=[2, 3, 0], predef_goal_pos=[8, 5, 0])
"""
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
    
   
    #Implementation of plotting functions in Graphs.py
 
    
    #g.plot_speed_time_2d(speeds, len(speeds))
    g.plot_movement_2d(ugv_pos, goal_pos, obstacles)
    g.plot_potential_field_surface(goal_pos, obstacles, field_size, res)
    
    # Allows for all plots to be shown at the same time
    plt.pause(0.001)
    plt.show(block = True)
    
"""