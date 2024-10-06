@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, goal_pos: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    temperature_position = 1.0
    temperature_velocity = 1.0
    
    # Calculate forward direction
    forward_vec = goal_pos - torso_position
    forward_vec[:, 2] = 0  # Ignore the z-axis for forward direction
    
    # Normalize forward vector
    forward_norm = torch.norm(forward_vec, p=2, dim=1, keepdim=True) + 1e-6  # Adding small value to avoid division by zero
    forward_direction = forward_vec / forward_norm  # Normalize
    
    # Calculate the cosine similarity between velocity and forward direction
    velocity_norm = torch.norm(velocity, p=2, dim=1, keepdim=True) + 1e-6
    velocity_normalized = velocity / velocity_norm  # Normalize velocity
    
    # Compute motion reward based on projection of velocity onto the forward direction
    motion_reward = torch.sum(forward_direction * velocity_normalized, dim=1)  # Will be between -1 and 1
    motion_reward = torch.exp(temperature_velocity * motion_reward) - 1  # Transforming for normalization

    # Distance penalty: further distance from goal decreases the reward
    distance_to_goal = torch.norm(forward_vec, p=2, dim=1)
    distance_reward = -distance_to_goal / 10.0  # Scale to make it a reasonable negative penalty
    distance_reward = torch.exp(temperature_position * distance_reward) - 1  # Transforming for normalization
    
    # Total reward
    total_reward = motion_reward + distance_reward

    # Creating reward components dictionary
    reward_components = {
        'motion_reward': motion_reward,
        'distance_reward': distance_reward,
    }

    return total_reward, reward_components
