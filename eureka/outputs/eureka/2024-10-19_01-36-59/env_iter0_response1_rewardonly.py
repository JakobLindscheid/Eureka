@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters
    temperature_linear = 0.1  # Temperature for linear velocity reward
    
    # Extract torso velocity in the forward direction (assumed to be along the x-axis)
    forward_velocity = root_states[:, 7]  # Replace index with correct one for forward motion

    # Calculate reward based on forward velocity
    linear_velocity_reward = torch.exp(forward_velocity)  # Exponential for normalization
    total_reward = linear_velocity_reward
    
    # Create a dictionary to hold individual reward components
    reward_components = {
        'linear_velocity_reward': linear_velocity_reward,
    }
    
    return total_reward, reward_components
