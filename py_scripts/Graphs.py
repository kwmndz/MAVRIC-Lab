import matplotlib
import matplotlib.pyplot as plt
from matplotlib import use
use('Agg') # so we can run on threads and force it to only write files and not try to display them (mostly for performance)
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from PotentialField import att_force, rep_force, K_ATT, K_REP, D_SAFE, OBSTACLE_HEIGHT
import plotly.graph_objects as go

"""

    Graphs for testing Potential Field Algorithm
    Both 2d and 3d graphs
    also some helper functions for graphing
    does not do anything if run directly

"""

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
def plot_potential_field_surface(goal_pos, obs_pos, field_size, res, log_dir, sim_id):
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
    
    plt.savefig(f'{log_dir}potential_field_{sim_id}.png')

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

# Plots the movement of UGV (X, Y) in 2D
def plot_movement_2d(ugv_pos, goal_pos, obstacles, log_dir, sim_id):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot UGV path
    ax.plot(ugv_pos[:, 0], ugv_pos[:, 1], '-o', label='UGV Path', color='blue')
    
    # Mark goal, start, and obstacles
    ax.scatter(goal_pos[0], goal_pos[1], color='black', label='Goal', s=100, marker='X')
    ax.scatter(ugv_pos[0, 0], ugv_pos[0, 1], color='green', label='Start', s=100, marker='o')
    
    # Ensure each obstacle is labeled once by using `label=''` after the first plot
    for idx, obs in enumerate(obstacles):
        label = 'Obstacle' if idx == 0 else ''  # Label only the first obstacle
        ax.scatter(obs[0], obs[1], color='red', s=60, label=label, marker='s')
    
    # Labeling
    ax.set_title('UGV Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    
    # Adjust layout and save directly to file
    plt.tight_layout()
    plt.savefig(f'{log_dir}movement_{sim_id}.png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    
# Plots the movement of UGV (X, Y) in 2D with an interactive plot
def plot_movement_interactive_2d(ugv_pos, goal_pos, obstacles, log_dir, sim_id , obs_container, et_local_min):
    # Create the plot
    fig = go.Figure()

    # Add the UGV path
    fig.add_trace(go.Scatter(x=ugv_pos[:,0], y=ugv_pos[:,1],
                             mode='lines+markers',
                             name='UGV Path',
                             marker=dict(size=5, color='blue'),
                             line=dict(width=2, color='blue')))
    
    # Add the goal position
    fig.add_trace(go.Scatter(x=[goal_pos[0]], y=[goal_pos[1]],
                             mode='markers',
                             name='Goal',
                             marker=dict(size=10, color='black')))

    # Add the starting position
    fig.add_trace(go.Scatter(x=[ugv_pos[0,0]], y=[ugv_pos[0,1]],
                             mode='markers',
                             name='Start',
                             marker=dict(size=10, color='green')))
    
    # Add the local minimum
    fig.add_trace(go.Scatter(x=[et_local_min[0]], y=[et_local_min[1]],
                                mode='markers',
                                name='Local Minimum',
                                marker=dict(size=10, color='black'))
                )

    # Add obstacles
    obs_x = [obs[0] for obs in obstacles]
    obs_y = [obs[1] for obs in obstacles]
    fig.add_trace(go.Scatter(x=obs_x, y=obs_y,
                             mode='markers',
                             name='Obstacles',
                             marker=dict(size=5, color='red')))

    # Set plot layout for better readability
    """fig.update_layout(title='UGV Path',
                      xaxis_title='X',
                      yaxis_title='Y',
                      width=800,
                      height=800,
                      showlegend=True)"""

    # Save the plot as an interactive HTML file
    #fig.write_html(f'{log_dir}/movement_{sim_id}.html')
    # Plot scanner trace
    for pos in ugv_pos:
        theta = np.linspace(-np.pi/2, np.pi/2, 100)  # 180 degrees in front of UGV
        x_scan = pos[0] + 5 * np.cos(theta)
        y_scan = pos[1] + 5 * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_scan, y=y_scan,
                                 mode='lines',
                                 name='Scanner Trace',
                                 line=dict(color='orange', width=1, dash='dash'),
                                 opacity=0.5))
        
    # Draw a dot for each point in obs_container and shade the inside red
    obs_container_x = obs_container[::2]
    obs_container_y = obs_container[1::2]
    width = 12.5
    
    obs_container_x_new = []
    obs_container_y_new = []
    
    direction_vector = np.array([obs_container_x[0] - obs_container_x[1], obs_container_y[0] - obs_container_y[1]])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize the direction vector
    tunnel_width = 12.5  # Define the width of the tunnel
    perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])  # Perpendicular to the direction vector
    
    # Calculate the new points for the obstacle container
    p1 = np.array([obs_container_x[0], obs_container_y[0]]) + perpendicular_vector * tunnel_width / 2
    p2 = np.array([obs_container_x[0], obs_container_y[0]]) - perpendicular_vector * tunnel_width / 2
    p3 = np.array([obs_container_x[1], obs_container_y[1]]) - perpendicular_vector * tunnel_width / 2
    p4 = np.array([obs_container_x[1], obs_container_y[1]]) + perpendicular_vector * tunnel_width / 2
                  
    obs_container_x_new.extend([p1[0], p2[0], p3[0], p4[0]])
    obs_container_y_new.extend([p1[1], p2[1], p3[1], p4[1]])
    
    # Add the dots
    fig.add_trace(go.Scatter(x=obs_container_x_new, y=obs_container_y_new,
                             mode='markers',
                             name='Obstacle Container Points',
                             marker=dict(size=5, color='purple')))
    
    # Shade the inside of the obstacle container
    fig.add_trace(go.Scatter(x=obs_container_x_new + [obs_container_x_new[0]], 
                             y=obs_container_y_new + [obs_container_y_new[0]],
                             fill='toself',
                             fillcolor='rgba(255, 0, 0, 0.2)',
                             line=dict(color='purple'),
                             name='Obstacle Container'))

    # Set plot layout for better readability
    fig.update_layout(title='UGV Path with Scanner Trace + Container',
                      xaxis_title='X',
                      yaxis_title='Y',
                      width=800,
                      height=800,
                      showlegend=True)

    # Save the plot as an interactive HTML file
    fig.write_html(f'{log_dir}/movement_{sim_id}.html')

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