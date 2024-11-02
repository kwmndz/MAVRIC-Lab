import numpy as np
from DynamicBayesianFiltering import SensorData

class Sensor:
    sensor_radius: float # Length of each ray
    sensor_angle: float # Angle about the x-axis dictating the field of view
    num_rays: int # Number of rays to send out

    def __init__(self, sensor_radius: float, sensor_angle: float, num_rays: int):
        self.sensor_radius = sensor_radius
        self.sensor_angle = sensor_angle
        self.num_rays = num_rays

    # Check if a ray sent from the current position collided with an obstacle
    # returns true if a collision happened, false otherwise
    def __check_collision(self, obstacles: np.ndarray, end_point: np.ndarray, pos_c: np.ndarray, collision_threshold = 1):
        # Number of steps to check along for collision
        # [5.1302186 4.0434062 0.
        num_steps = int(np.linalg.norm(end_point - pos_c[:-1]))
        for step in range(1, num_steps + 1):
            
            # Calc the point to check
            point_c = pos_c[:-1] + (end_point - pos_c[:-1]) * (step / num_steps)

            for obs in obstacles:
                if pos_c[0] - 5.1302186 <= 1e-2 and np.linalg.norm(point_c - obs[:-1]) < 1.7 and False:
                    print(np.linalg.norm(pos_c[:-1] - obs[:-1]))
                    #print("endpoint", np.linalg.norm(end_point - pos_c[:-1]))
                if np.linalg.norm(point_c - obs[:-1]) < collision_threshold:
                    return True
            
        return False

    # Scan the area around the current position to detect either occupied points or free space
    # returns a list of rays with the end point and whether a collision happened
    def scan_area(self, pos_c: np.ndarray, obstacles: np.ndarray):
        angle_increment = self.sensor_angle / self.num_rays
        rays = []

        cos_vals = np.cos(np.radians(np.linspace(-self.sensor_angle / 2, self.sensor_angle / 2, self.num_rays)))
        sin_vals = np.sin(np.radians(np.linspace(-self.sensor_angle / 2, self.sensor_angle / 2, self.num_rays)))

        for x, y in zip(cos_vals, sin_vals):
            direction = np.array([x, y])
            direction = direction / np.linalg.norm(direction)
            end_point = direction * self.sensor_radius + pos_c[:-1]

            collision = self.__check_collision(obstacles, end_point, pos_c)
            rays.append([end_point, collision])

        return rays
    
    # Returns the area scanned by the sensor
    def calc_scanned_area(self):
        return np.pi * self.sensor_radius**2 * self.sensor_angle / 360
    
    # Returns the AOI (Area of Interest)
    def calc_area_of_interest(self, pos_c: np.ndarray, local_min: np.ndarray):
        # Calculate the distance between the UGV and the local minimum
        distance = np.linalg.norm(local_min - pos_c)

        current_area = self.calc_scanned_area()
        area_from_local_min = np.pi * distance**2 * self.sensor_angle / 360

        return current_area + area_from_local_min
    
    # Returns  the obstacles in the recognized/scanned area
    def get_obstacles_in_recognized_area(self, pos_c: np.ndarray, obstacles: np.ndarray):
        
        seen_obstacles = []
        for obs in obstacles:
            if np.linalg.norm(obs - pos_c) < self.sensor_radius:
                seen_obstacles.append(obs)
        
        return np.array(seen_obstacles)
            
    # Returns the number of points in group A
    # "Group A comprises points between the initial local minimum and the UGV, including the local minimum itself" (from research paper)
    def get_group_a_points(self, pos_c: np.ndarray, local_min: np.ndarray, step_size = 0.1):
        distance = np.linalg.norm(local_min - pos_c)
        return int(np.ceil(distance / step_size)) # np.ceil automatically rounds up the number no matter the decimal value (e.g. 0.2 -> 1)
    
    # Returns true if the forces are parrallel, false otherwise
    def check_for_parrallel_forces(self, pos_c: np.ndarray, F_att: np.ndarray, F_rep: np.ndarray):
       
        # Check if the forces are parrallel
        cross_product = np.cross(F_att, F_rep)
        cross_product = cross_product[:-1]
        return np.allclose(cross_product, 0)
    
    # Returns the data from the sensor for the Dynamic Bayesian Filtering Algorithm
    # Returns in the form of SensorData from DynamicBayesianFiltering.py
    def get_sensor_data(self, pos_c: np.ndarray, local_min: np.ndarray, obstacles: np.ndarray):
        rays = self.scan_area(pos_c, obstacles)
        
        # Count the number of rays that detected a collision
        num_collisions = sum(1 for ray in rays if ray[1])
        recognized_area = self.calc_scanned_area()
        total_area = self.calc_area_of_interest(pos_c, local_min)
        
        return SensorData(num_collisions, len(rays), recognized_area, total_area)
        