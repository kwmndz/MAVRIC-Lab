import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SensorData:
    occupied_points: int
    total_points: int
    recognized_area: float
    total_area: float  # AOI
    
class DynamicBayesianFilter:
    def __init__(self, threshold: float = 0.85):
        """
        Initialize the Dynamic Bayesian Filter for local minima prediction.
        
        Args:
            threshold: Confidence threshold for local minimum prediction (default 0.85)
        """
        self.threshold = threshold
        self.prior_belief = None
        
    def initialize_belief(self, group_a_points: int) -> float:
        """
        Initialize belief based on number of potential local minima points.
        
        Args:
            group_a_points: Number of points in Group A (potential local minima points)
            
        Returns:
            Initial belief value
        """
        self.prior_belief = 1.0 / group_a_points
        return self.prior_belief
    
    def state_transition_probability(self, sensor_data: SensorData) -> float:
        """
        Calculate state transition probability P(X^t_lm | X^(t-1)_lm, u_t).
        
        Args:
            sensor_data: Current sensor readings
            
        Returns:
            State transition probability
        """
        # Calculate angle of occupied points (α/π)
        alpha = sensor_data.occupied_points / sensor_data.total_points
        return alpha / np.pi
    
    def observation_likelihood(self, sensor_data: SensorData) -> float:
        """
        Calculate observation likelihood P(z_t | X^t_lm = Local Min).
        
        Args:
            sensor_data: Current sensor readings
            
        Returns:
            Observation likelihood
        """
        # Calculate ratio of recognized area to total area of interest
        return sensor_data.recognized_area / sensor_data.total_area
    
    def normalize_belief(self, unnormalized_belief: float) -> float:
        """
        Normalize belief to ensure valid probability.
        
        Args:
            unnormalized_belief: Belief value before normalization
            
        Returns:
            Normalized belief value
        """
        # Simple normalization to keep probability between 0 and 1
        return max(0.0, min(1.0, unnormalized_belief))
    
    def update_belief(self, sensor_data: SensorData) -> Tuple[float, bool]:
        """
        Update belief using Dynamic Bayesian Filtering steps.
        
        Args:
            sensor_data: Current sensor readings
            
        Returns:
            Tuple of (updated belief, whether local minimum is predicted)
        """
        if self.prior_belief is None:
            raise ValueError("Must initialize belief before updating")
            
        # 1. Prediction Step
        transition_prob = self.state_transition_probability(sensor_data)
        
        # 2. Correction Step
        observation_prob = self.observation_likelihood(sensor_data)
        
        # 3. Combine steps and apply prior belief
        unnormalized_belief = observation_prob * transition_prob * self.prior_belief
        
        # 4. Normalize
        current_belief = self.normalize_belief(unnormalized_belief)
        
        # Update prior for next iteration
        self.prior_belief = current_belief
        
        # Check if belief exceeds threshold
        local_minimum_predicted = current_belief >= self.threshold
        
        return current_belief, local_minimum_predicted

def calculate_steps_to_minimum(current_pos: np.ndarray, 
                             predicted_minimum: np.ndarray, 
                             step_size: float = 1.0) -> int:
    """
    Calculate number of steps until UGV reaches predicted local minimum.
    
    Args:
        current_pos: Current UGV position [x, y]
        predicted_minimum: Predicted local minimum position [x, y]
        step_size: Size of each UGV step
        
    Returns:
        Estimated number of steps to local minimum
    """
    distance = np.linalg.norm(predicted_minimum - current_pos)
    return int(np.ceil(distance / step_size))

# Example usage:
def example_usage():
    # Initialize filter
    dbf = DynamicBayesianFilter(threshold=0.85)
    
    # Initialize belief with number of potential local minima points
    initial_belief = dbf.initialize_belief(group_a_points=10)
    
    # Simulate sensor data at each time step
    sensor_data = SensorData(
        occupied_points=5,
        total_points=20,
        recognized_area=50.0,
        total_area=100.0
    )
    
    # Update belief
    belief, is_minimum_predicted = dbf.update_belief(sensor_data)
    print(f"Updated belief: {belief}")
    
    if is_minimum_predicted:
        # Calculate steps to minimum if we have positions
        current_pos = np.array([0.0, 0.0])
        predicted_min = np.array([5.0, 5.0])
        steps = calculate_steps_to_minimum(current_pos, predicted_min)
        print(f"Local minimum predicted in {steps} steps with {belief:.2f} confidence")
        
if __name__ == "__main__":
    example_usage()