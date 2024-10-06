@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, sensor_forces: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters
    height_threshold = 0.1  # Threshold for the ball's height position (above the table)
    vel_threshold = 0.5  # Threshold for linear velocity (to minimize movement)
    sensor_force_threshold = 0.5  # Threshold for sensor force utilization

    # Compute rewards based on the ball's position
    ball_height = ball_positions[:, 2]  # Assuming z-axis is the height
    position_reward = torch.clamp(1.0 - (height_threshold - ball_height).abs() / height_threshold, min=0.0, max=1.0)

    # Compute reward based on the ball's linear velocity (we prefer low velocity)
    velocity_reward = torch.clamp(1.0 - (torch.norm(ball_linvels, dim=1) - vel_threshold) / vel_threshold, min=0.0, max=1.0)

    # Compute reward based on sensor forces (encourage usage of sensor forces on table)
    sensor_force_reward = torch.clamp(sensor_forces.mean(dim=1) / sensor_force_threshold, max=1.0)

    # Combine rewards
    total_reward = position_reward + velocity_reward + sensor_force_reward
    total_reward = torch.exp(total_reward)  # Normalize to positive values

    # Prepare individual reward components
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
        'sensor_force_reward': sensor_force_reward
    }

    return total_reward, reward_components
