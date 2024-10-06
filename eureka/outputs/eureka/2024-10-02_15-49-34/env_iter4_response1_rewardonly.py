@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso position and velocity
    torso_position = root_states[:, 0:3]  # Shape: (batch_size, 3)
    velocity = root_states[:, 7:10]  # Shape: (batch_size, 3)

    # Compute the x distance to the target
    to_target = targets - torso_position
    distance_to_target = to_target[:, 0]  # Only consider the x component for forward movement

    # Reward for reaching the target (linear reward for distance in x)
    distance_reward = torch.clamp(1.0 - torch.abs(distance_to_target), min=0.0)  # Reward for being close in x
    
    # Reward for forward speed (amplified)
    forward_speed_reward = 2.0 * velocity[:, 0]  # Forward speed, amplified to emphasize its importance
    
    # Combine rewards
    total_reward = distance_reward + forward_speed_reward
    
    # Temperature parameters
    temp_distance = 0.5
    temp_speed = 0.5
    
    # Apply temperature scaling
    distance_reward_transformed = torch.exp(distance_reward / temp_distance)
    forward_speed_reward_transformed = torch.exp(forward_speed_reward / temp_speed)

    # Total reward after transformation
    total_reward_transformed = distance_reward_transformed + forward_speed_reward_transformed
    
    # Create reward components dictionary
    reward_components = {
        'distance_reward': distance_reward,
        'forward_speed_reward': forward_speed_reward
    }
    
    return total_reward_transformed, reward_components
