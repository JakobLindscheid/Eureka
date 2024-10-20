@torch.jit.script
def compute_reward(velocity: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming it's along the x-axis)
    forward_velocity = velocity[:, 0]
    
    # Reward component for high forward velocity
    velocity_reward = forward_velocity
    
    # Penalize high angular velocities to encourage stable running
    ang_velocity_penalty = torch.sum(torch.abs(ang_velocity), dim=-1)
    
    # Normalize these components and combine them
    velocity_temperature = 0.1  # temperature parameter for transforming velocity reward
    ang_velocity_temperature = 0.1  # temperature parameter for transforming angular velocity penalty

    # Apply a transformation with temperature parameters
    transformed_velocity_reward = torch.exp(velocity_reward / velocity_temperature)
    transformed_ang_velocity_penalty = torch.exp(-ang_velocity_penalty / ang_velocity_temperature)

    # Total reward is the sum of the transformed components
    total_reward = transformed_velocity_reward + transformed_ang_velocity_penalty

    # Return the total reward and individual components
    reward_components = {
        "velocity_reward": transformed_velocity_reward,
        "angular_velocity_penalty": transformed_ang_velocity_penalty
    }

    return total_reward, reward_components
