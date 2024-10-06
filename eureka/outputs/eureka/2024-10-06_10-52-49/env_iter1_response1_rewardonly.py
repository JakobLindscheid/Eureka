@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.5  # Adjusting temperature for speed for better scaling

    # Extracting torso velocity (x, y components) for horizontal speed calculation
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Calculate the horizontal speed of the torso
    horizontal_speed = torch.norm(torso_velocity[:, :2], p=2, dim=-1)  # Considering x and y components only for running
    
    # Normalize the speed reward using the exponential function
    reward = torch.exp(horizontal_speed / speed_temp)  # Maintain a strong signal for large speeds
    
    # Create individual reward components for analysis/monitoring
    reward_components = {
        'horizontal_speed': horizontal_speed,
    }
    
    return reward.sum(), reward_components  # Return total reward and individual components
