import numpy as np
import Graphs as g
import matplotlib.pyplot as plt

# Constants for Potential Field Algorithm
K_ATT = 1.0 # Attractive
K_REP = 0.5 # Repulsive
D_SAFE = 2 # Safe distance

# Constants for testing simulation (keep low or potential field algorithm gives weird results)
STEP_SIZE = 0.001
OBSTACLE_HEIGHT = 10 # max height of obstacles in field graph


"""

    Potential Field Algorithm Calculations
    https://www.youtube.com/watch?v=Ls8EBoG_SEQ
    Works for 3D and 2D
     
"""


# Returns attractive force
# Assuming goal_pos and ugv_pos are vectors
# F_att = K_att * (goal_pos - ugv_pos)
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


# Testing the simulation
if __name__ == '__main__':
    # testing constants
    pos_i = np.array([2, 3, 0])
    goal_pos = np.array([8, 8, 0])
    obstacles = [[7, y, 0] for y in range(3, 8)]
    
    """
    midpoint = (pos_i + goal_pos) / 2 # midpoint between start and goal
    obstacles.append([midpoint[0], midpoint[1], midpoint[2]])
    """
    
    # amt of steps to simulate
    num_steps = 100000
    
    # constants for potential field surface graph
    field_size = 10
    res = 100 # higher res = more detail but more lag
    
    ugv_pos, speeds = sim_movement(pos_i, goal_pos, obstacles, num_steps)
    
    """
    Implementation of plotting functions in Graphs.py
    """
    
    #g.plot_speed_time_2d(speeds, len(speeds))
    g.plot_movement_2d(ugv_pos, goal_pos, obstacles)
    g.plot_potential_field_surface(goal_pos, obstacles, field_size, res)
    
    # Allows for all plots to be shown at the same time
    plt.pause(0.001)
    plt.show(block = True)