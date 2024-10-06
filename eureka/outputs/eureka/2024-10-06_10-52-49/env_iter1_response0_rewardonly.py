@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, episode_length: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.5  # Lowered to enhance sensitivity on speed rewards
    length_temp = 0.01  # Scaling factor for the episodic length reward

    # Extract the torso velocity (x, y, z components)
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Calculate the speed of the torso (magnitude of the velocity vector)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Shape: (N,)
    
    # Reward is based on the speed, normalized with an exponential function
    speed_reward = torch.exp(speed / speed_temp)  # Larger rewards for higher speeds

    # Reward based on the episode length (positive impact on longer successful episodes)
    length_reward = episode_length * length_temp  # Directly scaling the episode length
    
    # Total reward is a combination of the speed and length rewards
    total_reward = speed_reward + length_reward

    # Create individual reward components for monitoring
    reward_components = {
        'speed': speed_reward,
        'length': length_reward,
    }
    
    return total_reward.sum(), reward_components  # Return total reward and individual components
