@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    velocity_temp = 1.0

    # Extract the torso velocity (x, y, z components)
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Calculate the speed of the torso (magnitude of the velocity vector)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Shape: (N,)

    # Reward is based on the speed, we normalize it using the exponential function
    reward = torch.exp(speed / velocity_temp)  # Reward becomes larger for higher speeds
    
    # Create individual reward components for monitoring
    reward_components = {
        'speed': speed,
    }
    
    return reward.sum(), reward_components  # Return total reward and individual components
