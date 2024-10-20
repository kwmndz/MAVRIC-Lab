import numpy as np

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
    def __check_collision(obstacles: np.ndarray, end_point: np.ndarray, pos_c: np.ndarray, collision_threshold = 0.001):
        # Number of steps to check along for collision
        num_steps = int(np.linalg.norm(end_point - pos_c))

        for step in range(num_steps + 1):
            
            # Calc the point to check
            point_c = pos_c + (end_point - pos_c) * (step / num_steps)

            for obs in obstacles:
                if np.linalg.norm(point_c - obs) < collision_threshold:
                    return True
            
        return False

    # Scan the area around the current position to detect either occupied points or free space
    # returns a list of rays with the end point and whether a collision happened
    def scan_area(self, pos_c: np.ndarray, obstacles: np.ndarray):
        # Calc the angle between each ray
        angle_increment = self.sensor_angle / self.num_rays

        rays = []

        for i in range(self.num_rays):
            # Calc the angle of the ray
            ray_angle = -self.sensor_angle / 2 + i * angle_increment

            # Calc the x and y coords of the end point of the ray
            x = self.sensor_radius * np.cos(np.radians(ray_angle)) + pos_c[0]
            y = self.sensor_radius * np.sin(np.radians(ray_angle)) + pos_c[1]

            end_point = np.array([x, y])
            collision = self.__check_collision(obstacles, end_point, pos_c) # Check for collision

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
    
    # Returns the number of points in group A
    # "Group A comprises points between the initial local minimum and the UGV, including the local minimum itself" (from research paper)
    def get_group_a_points(self, pos_c: np.ndarray, local_min: np.ndarray, step_size = 0.5):
        distance = np.linalg.norm(local_min - pos_c)
        return int(np.ceil(distance / step_size)) # np.ceil automatically rounds up the number no matter the decimal value (e.g. 0.2 -> 1)
        