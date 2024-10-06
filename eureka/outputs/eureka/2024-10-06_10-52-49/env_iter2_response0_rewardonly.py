@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.3  # Lower temperature to increase reward sensitivity
    distance_temp = 0.1
    
    # Extracting torso velocity (velocity of the humanoid)
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Calculate speed (magnitude of the velocity)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # (N,)
    speed_reward = torch.exp(speed / speed_temp)  # Exponential to normalize speed
    
    # Calculate distance to target (as positive reward, shorter distance means better)
    torso_position = root_states[:, 0:3]  # Shape: (N, 3)
    to_target = targets - torso_position  # Target vector
    distance = torch.norm(to_target, p=2, dim=-1)  # (N,)
    
    # Reward based on distance (the smaller, the better)
    distance_reward = torch.exp(-distance / distance_temp)  # Exponentially incentivizing closeness to the target

    # Combined total reward 
    total_reward = speed_reward + distance_reward  # Encourages both speed and proximity

    # Create a dictionary to return individual components for analytics
    reward_components = {
        'speed_reward': speed_reward,
        'distance_reward': distance_reward,
    }
    
    return total_reward.sum(), reward_components  # Return total and individual rewards
