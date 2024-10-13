import numpy as np
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
    Also plots the data
    For testing purposes
    2d and 3d simulations/graphs
    
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

# helper function for plot_potential_field_surface
# returns the vectors of the potential field
def calc_potential_vectors(X, Y, goal_pos, obs_pos):
    
    pos = np.stack((X, Y), axis=-1)
    
    F_att = 0.5 * K_ATT * np.sum((pos - goal_pos[:2])**2, axis=-1)
    F_rep = np.zeros_like(X)
    
    for obs in obs_pos:
        distance = np.linalg.norm(pos - obs[:2], axis=-1)
        F_rep += np.where(distance < D_SAFE, 
                          K_REP * (1/distance - 1/D_SAFE)**2, 
                          K_REP * np.exp(-distance + D_SAFE) * OBSTACLE_HEIGHT /10)

    return F_att + F_rep

# Plot the pontential field surface (like they did in the video)
def plot_potential_field_surface(goal_pos, obs_pos, field_size, res = 50):
    x = np.linspace(0, field_size, res)
    y = np.linspace(0, field_size, res)
    X, Y = np.meshgrid(x, y)
    
    Z = calc_potential_vectors(X, Y, goal_pos, obs_pos)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # reduce resolution for less lag when moving plot around
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none',
                           rstride = 2, cstride = 2, linewidth=0, antialiased=False)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.scatter(*goal_pos, color='green', label='Goal', s=100)
    for obs in obs_pos:
        ax.scatter(obs[0], obs[1], OBSTACLE_HEIGHT, color='red', label='Obstacle', s=100)
    
    # Labeling
    ax.set_title('Potential Field')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential')
    ax.legend()
    
    plt.show(block = False)

# Plots the movement of UGV in 3D
def plot_movement_3d(ugv_pos, goal_pos, obstacles):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    
    ugv_pos = np.array(ugv_pos)
    ax.plot(ugv_pos[:,0], ugv_pos[:,1], ugv_pos[:,2], '-o', label='UGV Path')
    
    # Mark the goal and obsticles on the plot
    ax.scatter(*goal_pos, color='black', label='Goal', s=100)
    for obs in obstacles:
        ax.scatter(*obs, color='red', label='Obstacle', s=100)
        
    # Labeling
    ax.set_title('UGV Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)
    
    plt.show(block = False)

# Plots teh movement of UGV (X, Y) in 2D
def plot_movement_2d(ugv_pos, goal_pos, obstacles):
    plt.figure(figsize=(6,6))
    plt.plot(ugv_pos[:,0], ugv_pos[:,1], '-o', label='UGV Path')
    
    # Mark the goal and obsticles on the plot
    plt.scatter(goal_pos[0], goal_pos[1], color='black', label='Goal', s=100)
    for obs in obstacles:
        plt.scatter(obs[0], obs[1], color='red', label='Obstacle', s=100)
        
    # Labeling
    plt.title('UGV Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    plt.show(block = False)
    
# Plots the speed vs time of the UGV
def plot_speed_time_2d(speeds, num_steps):
    plt.figure(figsize=(6,6))
    plt.plot(range(num_steps), speeds, '-o')
    
    # Labeling
    plt.title('UGV Speed vs Time')
    plt.xlabel('Time Step')
    plt.ylabel('Speed')
    plt.grid(True)
    
    plt.show(block = False)
    
# plot the entire gradient of the potential field
def plot_field_gradient(goal_pos, obs_pos, field_size, res = 20):
    x = np.linspace(0, field_size, res)
    y = np.linspace(0, field_size, res)
    X, Y = np.meshgrid(x, y)
    
    # potential arrays
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(res):
        for j in range(res):
            pos = [X[i,j], Y[i,j]]
            F_att = att_force(pos, goal_pos)
            F_rep = np.array([0.0, 0.0])
            for obs in obs_pos:
                F_rep += rep_force(pos, obs)
            
            F_resultant = F_att + F_rep
            U[i,j] = F_resultant[0]/10
            V[i,j] = F_resultant[1]/10
    
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V, color='blue', scale=50)
    plt.plot(goal_pos[0], goal_pos[1], 'go', label="Goal")  
    
    for obs in obs_pos:
        plt.plot(obs[0], obs[1], 'ro', label="Obstacle")  
        
    plt.title("Potential Field Gradient")
    plt.xlim(0, field_size)
    plt.ylim(0, field_size)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    
    # so the plots dont spawn on top of eachother
    manager = plt.get_current_fig_manager()
    manager.window.setGeometry(800, 100, 800, 800)  
    
    plt.show(block = False)


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
    res = 50 # higher res = more detail but more lag
    
    ugv_pos, speeds = sim_movement(pos_i, goal_pos, obstacles, num_steps)
    
    
    #plot_speed_time_2d(speeds, len(speeds))
    plot_movement_2d(ugv_pos, goal_pos, obstacles)
    #plot_field_gradient(goal_pos, obstacles, 15, 100)
    plot_potential_field_surface(goal_pos, obstacles, field_size, res)
    
    # Allows for all plots to be shown at the same time
    plt.pause(0.001)
    plt.show(block = True)