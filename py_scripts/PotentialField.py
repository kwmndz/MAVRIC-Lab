import numpy as np
import matplotlib.pyplot as plt

# Constants for Potential Field Algorithm
K_ATT = 1.0 # Attractive
K_REP = 0.5 # Repulsive
D_SAFE = 2 # Safe distance

# Constants for testing simulation (prabably not needed)
STEP_SIZE = 0.1


"""

    Potential Field Algorithm Calculations
    2D Vectors only right now
    https://www.youtube.com/watch?v=Ls8EBoG_SEQ
     
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
        return np.array([0.0, 0.0])
    
    # unit vector of direction
    direction = (np.array(ugv_pos) - np.array(obs_pos)) / distance
    return K_REP * (1/distance - 1/D_SAFE) * (1/(distance**2)) * direction


"""

    Simulation of movement using Potential Field Algorithm
    Also plots the data
    For testing purposes
    Only working with 2D vectors
    
"""


def sim_movement_2d(pos_i, goal_pos, obstacles, num_steps):
    ugv_pos = [pos_i]
    speeds = []
    
    for _ in range(num_steps):
        pos_c = ugv_pos[-1] # Always get last added position
        
        # Potential Field Algorithm for force calcs
        F_att = att_force(pos_c, goal_pos)
        F_rep = np.array([0.0, 0.0])
        for obs in obstacles:
            F_rep += rep_force(pos_c, obs)
        
        F_resultant = F_att + F_rep
        speed = np.linalg.norm(F_resultant)
        speeds.append(speed)
        
        # Update position
        pos_n = pos_c + F_resultant * STEP_SIZE
        ugv_pos.append(pos_n)
        
    return np.array(ugv_pos), np.array(speeds)

# Plots teh movement of UGV (X, Y) in 2D
def plot_movement_2d(ugv_pos, goal_pos, obstacles):
    plt.figure(figsize=(6,6))
    plt.plot(ugv_pos[:,0], ugv_pos[:,1], '-o', label='UGV Path')
    
    # Mark the goal and obsticles on the plot
    plt.scatter(*goal_pos, color='black', label='Goal', s=100)
    for obs in obstacles:
        plt.scatter(*obs, color='red', label='Obstacle', s=100)
        
    # Labeling
    plt.title('UGV Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
# Plots the speed vs time of the UGV
def plot_speed_time_2d(speeds, num_steps):
    plt.figure(figsize=(6,6))
    plt.plot(range(num_steps), speeds, '-o')
    
    # Labeling
    plt.title('UGV Speed vs Time')
    plt.xlabel('Time Step')
    plt.ylabel('Speed')
    plt.grid(True)
    
    plt.show()


# Testing the simulation
if __name__ == '__main__':
    # testing constants
    pos_i = np.array([2, 3])
    goal_pos = np.array([10, 10])
    obstacles = [[5, 5], [7, 8]]
    
    # amt of steps to simulate
    num_steps = 100
    
    ugv_pos, speeds = sim_movement_2d(pos_i, goal_pos, obstacles, num_steps)
    
    plot_movement_2d(ugv_pos, goal_pos, obstacles)
    plot_speed_time_2d(speeds, num_steps)