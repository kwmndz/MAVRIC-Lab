import numpy as np
from dataclasses import dataclass

@dataclass
class SensorData:
    occupied_points: int
    total_points: int
    recognized_area: float # RA(X_t)
    total_area: float # AOI
    
    def __str__(self):
        return (f"SensorData(occupied_points={self.occupied_points}, "
                f"total_points={self.total_points}, "
                f"recognized_area={self.recognized_area}, "
                f"total_area={self.total_area})")

# Dynamic Bayesian Filter for local minima prediction
class DBF:
    belief_prior: float
    threshold: float # Confidence threshold

    def __init__(self, threshold = 0.85):
        self.threshold = threshold
        self.belief_prior = None

    # Initialize belief based on number of potential local minima points
    # "Group A comprises points between the initial local minimum and the UGV" (from research paper)
    def initialize_belief(self, group_a_points):
        self.belief_prior = 1.0 / group_a_points
        return self.belief_prior
    
    # Calculate state transition probability P(X^t_lm | X^(t-1)_lm, u_t)
    # sensor_data is of type SensorData
    # alpha represents "the angle of occupied points which refers to the ratio of the number of occupied points to the total number of sensor points." (from research paper)
    # Prediction Step
    def state_transition_probability(self, sensor_data):
        alpha = sensor_data.occupied_points / sensor_data.total_points
        return alpha + 1
    
    # Calculate observation likelihood P(z_t | X^t_lm = Local Min)
    # sensor_data is of type SensorData
    # Correction Step
    def observation_likelihood(self, sensor_data):
        return sensor_data.recognized_area / sensor_data.total_area
    
    # Normalize belief value to be within [0, 1]
    # Normalization step
    def normalize(self, belief):
        return max(0.0, min(1.0, belief))
    
    # Update belief of local minimum based on sensor data
    # Combination of functions above
    # sensor_data is of type SensorData
    # steps in this function found through the psuedocode in the research paper
    def update_belief(self, sensor_data):
        log_str = ""
        if self.belief_prior is None:
            raise ValueError("Remeber to initialize the belief first (initialize_belief function)")
        
        prediction = self.state_transition_probability(sensor_data)
        correction = self.observation_likelihood(sensor_data)

        posterior = prediction * correction * self.belief_prior 
        #print(f"Prediction: {prediction}, Correction: {correction}, Prior: {self.belief_prior}, Posterior: {posterior}")
        log_str += f"Prediction: {prediction}, Correction: {correction}, Prior: {self.belief_prior}, Posterior: {posterior}"
        
        belief_current = self.normalize(posterior)
        self.belief_prior = belief_current
        #print (f"Belief: {belief_current}")
        log_str += f" Belief after normalization: {belief_current}"
        # returns the current belieft and True if the belief is greater than the threshold, false otherwise
        return belief_current, belief_current >= self.threshold, log_str

# Calculate the number of steps to reach the local minimum
# pos_c & min_predicted are vectors stored in numpy arrays
def steps_to_local_min(pos_c:np.ndarray, min_predicted: np.ndarray, step_size = 0.1):

    # "The estimated number of steps before the UGV reaches X_lm can be calculated by dividing the distance." (from research paper)
    distance = np.linalg.norm(min_predicted - pos_c)
    return int(np.ceil(distance / step_size)) # np.ceil automatically rounds up the number no matter the decimal value (e.g. 0.2 -> 1)

