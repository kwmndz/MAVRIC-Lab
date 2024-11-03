import numpy as np
import Graphs as g
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Sensor import Sensor
from Sensor import get_obstacles_in_recognized_area_optimized
from DynamicBayesianFiltering import DBF, SensorData, steps_to_local_min
import sys
import os
import csv
import concurrent.futures
import time
from numba import njit

# Constants for Potential Field Algorithm
K_ATT = 1.0 # Attractive
K_REP = 0.5 # Repulsive
D_SAFE = 5 # Safe distance / Scanner Radius

# Constants for testing simulation (keep low or potential field algorithm gives weird results)
STEP_SIZE = 0.01
OBSTACLE_HEIGHT = 10 # max height of obstacles in field graph
MAX_VEL = 1.0 # max velocity of UGV


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

# Optimized version of rep_force using numpy for vectorized operations
@njit
def rep_force_optimized(ugv_pos, obstacles):
    # Convert to numpy arrays for vectorized operations
    #ugv_pos = np.array(ugv_pos)
    #obstacles = np.array(obstacles)
    # Calculate displacement vectors and distances
    displacement_vectors = obstacles - ugv_pos
    #print(displacement_vectors)
    distances = np.sqrt(np.sum(displacement_vectors**2, axis=1))
    #print(len(distances))
    #print(len(displacement_vectors))
    #print(ugv_pos)
    # --- obstacles we are looking at are all within range alr ---
    # Filter obstacles within D_SAFE distance
    # within_safe_dist = distances < D_SAFE --- obstacles we are looking at are all within range alr
    #displacement_vectors = displacement_vectors[distances]
    #distances = distances[within_safe_dist]
    
    # If there are no obstacles within the safe distance, return zero force
    """
    if distances.size == 0:
        print(ugv_pos)
        return np.zeros_like(ugv_pos)"""

    # Calculate unit direction vectors
    epsilon = 1e-25
    directions = -1 * displacement_vectors / (distances[:, np.newaxis] + epsilon)

    # Compute repulsive forces for each obstacle
    forces = K_REP * ( (1/(distances[:, np.newaxis] + epsilon) - 1/D_SAFE) * (1/(distances[:, np.newaxis] + epsilon)**2) ) * directions
   # print(forces[1])

    # Sum up all repulsive forces
    total_force = np.sum(forces, axis=0)
    #print(total_force)
    return total_force

# Checks if the ugv is at a local min
def is_local_min(force_net, threshold=1e-3):
    force_mag = np.linalg.norm(force_net)
    
    # If force is below small threshold, then it is a local min
    return force_mag < threshold

# helper method to calculate the force difference 
def force_difference(ugv_pos, goal_pos, obstacles):
    F_att = att_force(ugv_pos, goal_pos) 
    if F_att[0] < 1e-6 and F_att[1] < 1e-6:
        return 10   
    
    #F_rep =np.sum([rep_force(ugv_pos, obs) for obs in obstacles], axis=0) 
    F_rep = rep_force_optimized(ugv_pos, obstacles) if obstacles.size > 0 else np.zeros(len(ugv_pos))
    
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
    """
    removed due to performance reseaons
    F_att = att_force(result.x, goal_pos)
    F_rep = np.zeros(len(result.x))
    for obs in obstacles:
        F_rep += rep_force(result.x, obs)
    """
    
    return {
        'local_min': result.x,
        'F_att': 0,
        'F_rep': 0,
        'F_net': 0 + 0,
        'force_mag': result.fun,
        'success': result.success,
        'iterations': result.nit
    }


"""

    Simulation of movement using Potential Field Algorithm
    For testing purposes
    
"""
# Helper function to generate clustered obstacles
# Clusters the points together so obstacles are more realistic and not just points
def generate_obstacles(num_obstacles, step_size=0.001, length_range=(5, 20), area_size=[100, 100, 0, 0]):
    obstacles = []
    # Randomly decide the position of the starting point of the obstacle
    # Calculate the direction vector from (area_size[2], area_size[3]) to (area_size[0], area_size[1])
    direction_vector = np.array([area_size[0] - area_size[2], area_size[1] - area_size[3]])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize the direction vector
    # Add a random width offset perpendicular to the tunnel direction
    tunnel_width = 12.5  # Define the width of the tunnel
    perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])  # Perpendicular to the direction vector
    
    
    for _ in range(num_obstacles):
        
    
        # Generate a random distance along the tunnel
        distance_along_tunnel = np.random.uniform(0, np.linalg.norm([area_size[0] - area_size[2], area_size[1] - area_size[3]]))
        # Calculate the starting point based on the distance along the tunnel
        start_x = area_size[2] + distance_along_tunnel * direction_vector[0]
        start_y = area_size[3] + distance_along_tunnel * direction_vector[1]
        
        width_offset = np.random.uniform(-tunnel_width / 2, tunnel_width / 2)
        
        start_x += width_offset * perpendicular_vector[0]
        start_y += width_offset * perpendicular_vector[1]
        
        # Ensure the obstacle is within the tunnel boundaries
        start_x = np.clip(start_x, area_size[2], area_size[0])
        start_y = np.clip(start_y, area_size[3], area_size[1])
        
        
        # Randomly decide the orientation and length of the obstacle
        angle = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
        length = np.random.uniform(length_range[0], length_range[1])  # Length of the obstacle line
        
        # Generate points along the line based on the starting point, angle, and step size
        num_points = int(length / step_size) + 1  # Calculate the number of points based on length and step size
        for i in range(num_points):
            offset = step_size * i  # Distance between points along the line
            x = start_x + offset * np.cos(angle)
            y = start_y + offset * np.sin(angle)
            obstacles.append(np.array([x, y, 0]))
    
    return np.array(obstacles)

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
    estimated_local_min = None
    prior_estimated_local_min = None
    
    for _ in range(num_steps):
        pos_c = ugv_pos[-1] # Always get last added position
        
        #obstacles_in_range = #sensor.get_obstacles_in_recognized_area(pos_c, obstacles)
        obstacles_in_range = get_obstacles_in_recognized_area_optimized(pos_c, obstacles, sensor.sensor_radius)
        obstacles_in_range = np.array(obstacles_in_range)
        
        # Potential Field Algorithm for force calcs
        F_att = att_force(pos_c, goal_pos)
        
        #print(len(pos_c))
        F_resultant = F_att
            
        #F_rep = np.sum([rep_force(pos_c, obs) for obs in obstacles_in_range], axis=0)
        if not obstacles_in_range.size == 0:
            #print("prre", len(pos_c), len(obstacles_in_range[0]))
            F_rep = rep_force_optimized(pos_c, obstacles_in_range) # optimized version
            #F_rep = np.sum([rep_force(pos_c, obs) for obs in obstacles_in_range], axis=0)
            #print("normal func:")
            #print(np.sum([rep_force(pos_c, obs) for obs in obstacles_in_range], axis=0))
            #print(len(F_rep))
            F_resultant += F_rep
        else:
            F_rep = np.zeros(len(pos_c))
        
        
        
        if sensor.check_for_parrallel_forces(pos_c, F_att, F_rep) and np.linalg.norm(F_rep) > 1e-8:
            if prior_estimated_local_min is None:
                estimated_local_min = find_potential_local_min(pos_c, goal_pos, obstacles_in_range)
            else:
                estimated_local_min = find_potential_local_min(pos_c, goal_pos, obstacles_in_range, guess_i=prior_estimated_local_min['local_min'])
            #print("herer" , estimated_local_min, prior_estimated_local_min, first_check)
            if not first_check and (estimated_local_min['local_min'][0] - prior_estimated_local_min['local_min'][0] > 0.01 or estimated_local_min['local_min'][0] - prior_estimated_local_min['local_min'][1] > 0.01):
                first_check = True
                #print(len(obstacles_in_range))
                #previus_num_obs = len(obstacles_in_range)
            elif not first_check:
                estimated_local_min = prior_estimated_local_min
            if first_check:
                group_a_points = sensor.get_group_a_points(pos_c, estimated_local_min['local_min'], step_size = 0.1)
                #print(dbf.initialize_belief(group_a_points))
                log_string += f"Initial belief: {dbf.initialize_belief(group_a_points)}\n"
                first_check = False
                prior_estimated_local_min = estimated_local_min
                
            sensorData, log_str1 = sensor.get_sensor_data(pos_c, estimated_local_min['local_min'], obstacles_in_range)
            #print(sensorData)
            #print(pos_c)
            #print(obstacles_in_range)
            belief, reached_threshold, log_str2 = dbf.update_belief(sensorData)
            
            log_entries = []
            if belief > 0.1:
                log_entries.append(f"{_} Step")
                log_entries.append(log_str2)
                log_entries.append(log_str1)
                log_entries.append(f"{estimated_local_min['local_min']} Estimated Local Min")
                log_entries.append(f"{estimated_local_min['force_mag']} Force Magnitude")
                log_entries.append(f"{estimated_local_min['iterations']} Iterations")
                log_entries.append(f"{estimated_local_min['success']} Success")
                #log_entries.append(f"{estimated_local_min['F_att']} F_att")
                #log_entries.append(f"{estimated_local_min['F_rep']} F_rep")
                log_entries.append(f"{len(obstacles_in_range)} Obstacles in Range")
                log_entries.append(f"{belief} Belief\n")
            else:
                log_entries.append(f"{_} Step")
                log_entries.append(log_str2)
                log_entries.append(log_str1)
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
        vel = F_resultant * STEP_SIZE
        if np.linalg.norm(vel) > MAX_VEL:
            vel = vel / np.linalg.norm(vel) * MAX_VEL
        pos_n = pos_c + vel
        ugv_pos.append(pos_n)
        
        if np.linalg.norm(pos_n - goal_pos) < 0.75:
            log_string += f'Goal Reached, step: {_}\n'
            break
        
        if is_local_min(F_resultant):
            log_string += f'Local Min Found, step: {_}\n'
            log_string += f"{pos_n} UGV Position\n"
            log_string += f"Wasnt predicted :( (bad) \n)"
            break
        
    return np.array(ugv_pos), np.array(speeds), log_string, estimated_local_min['local_min']

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
        num_obstacles = np.random.randint(grid_size**2 // 100, grid_size**2 // 50)
        #obstacles = [np.array([np.random.uniform(0, grid_size), np.random.uniform(0, grid_size), 0]) for _ in range(num_obstacles)]
        obstacles = generate_obstacles(12, step_size=0.05, length_range=(5, 10), area_size=[goal_pos[0] - 2.5, goal_pos[1] - 2.5, pos_i[0] + 2.5, pos_i[1] + 2.5])

    
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
    ugv_pos, speeds, log_string, et_local_min = sim_movement_with_DBF(pos_i, goal_pos, obstacles, num_steps=10000)
    
    # Close log file and reset stdout
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    
    # Plot and save plots to log directory
    #g.plot_movement_2d(ugv_pos, goal_pos, obstacles, log_dir, sim_id)
    g.plot_movement_interactive_2d(ugv_pos, goal_pos, obstacles, log_dir, sim_id, [goal_pos[0] - 2.5, goal_pos[1] - 2.5, pos_i[0] + 2.5, pos_i[1] + 2.5], et_local_min)
    
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
    run_multiple_simulations(15, 100, predef_ugv_pos=[2, np.random.uniform(5, 30), 0], predef_goal_pos=[92, np.random.uniform(50, 98), 0])
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