import numpy as np

def check_collision(obs: np.ndarray, end_point: np.ndarray, pos_c: np.ndarray, collision_threshold = 0.001):
    # Calculate the number of steps for the interpolation
    num_steps = int(np.linalg.norm(end_point - pos_c))

    # Interpolate points along the path
    for step in range(num_steps + 1):
        # Calculate the interpolated point
        interp_point = pos_c + (end_point - pos_c) * (step / num_steps)

        if np.linalg.norm(interp_point - obs) < collision_threshold:
            return True
        
    return False



def scan_area(radius, angle, num_rays, pos_c: np.ndarray):
    # Calculate the angle between each ray
    angle_increment = angle / num_rays

    # Initialize list to store ray distances
    ray_endpoints = []

    # Loop through each ray
    for i in range(num_rays):
        # Calculate the angle of the ray
        ray_angle = -angle / 2 + i * angle_increment

        # Calculate the x and y coordinates of the end point of the ray
        x = radius * np.cos(np.radians(ray_angle)) + pos_c[0]
        y = radius * np.sin(np.radians(ray_angle)) + pos_c[1]

        ray_endpoints.append((x, y))

    return ray_endpoints