@torch.jit.script
def compute_reward(root_positions: torch.Tensor, 
                   root_linvels: torch.Tensor, 
                   target_height: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Define temperature parameters for reward transformations
    position_temp = 1.0
    velocity_temp = 1.0

    # Compute the distances from the target position
    position_error = torch.abs(root_positions[..., 2] - target_height)  # Distance to target height
    distance_reward = torch.exp(-position_temp * position_error)  # Reward for being close to target height
    
    # Compute the linear velocity
    velocity_magnitude = torch.norm(root_linvels, dim=-1)
    velocity_reward = torch.exp(-velocity_temp * velocity_magnitude)  # Penalty for moving too quickly

    # Total reward is a combination of the distance and velocity rewards
    total_reward = distance_reward + velocity_reward

    # Create the reward component dictionary
    reward_components = {
        'distance_reward': distance_reward,
        'velocity_reward': velocity_reward,
    }
    
    return total_reward, reward_components
