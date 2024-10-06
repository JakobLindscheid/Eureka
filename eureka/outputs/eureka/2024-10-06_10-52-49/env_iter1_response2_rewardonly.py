@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, episode_length: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.5
    length_temp = 0.1

    # Extract the torso velocity (x, y, z components)
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Calculate the speed of the torso (magnitude of the velocity vector)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Shape: (N,)
    
    # Reward based on the square of the speed
    speed_reward = speed ** 2 / speed_temp

    # Negative reward based on episode length to encourage shorter episodes
    length_penalty = -episode_length / length_temp
   
    # Total reward aggregating speed reward and length penalty
    reward = speed_reward + length_penalty

    # Create individual reward components for monitoring
    reward_components = {
        'speed': speed,
        'length_penalty': length_penalty,
    }
    
    return reward.sum(), reward_components  # Return total reward and individual components
