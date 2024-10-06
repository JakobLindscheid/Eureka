@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, 
                   sensor_forces: torch.Tensor, sensor_torques: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    reward_temperature = 0.1  # Temperature for scaling reward components
    position_penalty_temperature = 0.1  # Temperature for position penalty
    velocity_penalty_temperature = 0.1  # Temperature for velocity penalty
    force_penalty_temperature = 0.1  # Temperature for sensor force penalty
    torque_penalty_temperature = 0.1  # Temperature for sensor torque penalty

    # Reward Components
    on_table = (ball_positions[..., 2] > 0).float()  # Reward for being above the table (z > 0)
    height_reward = on_table * 10.0  # Higher reward for being above the table
    height_reward = height_reward / (1 + torch.exp(-reward_temperature * (ball_positions[..., 2])))

    falling_penalty = (1 - on_table) * (-1.0)  # Penalty for falling
    falling_penalty = falling_penalty / (1 + torch.exp(-position_penalty_temperature * (1 - ball_positions[..., 2])))

    # Velocity penalty (we want the ball to be stationary)
    stationary_penalty = -torch.norm(ball_linvels, dim=-1)  # Negative reward for velocity
    stationary_penalty = stationary_penalty / (1 + torch.exp(-velocity_penalty_temperature * torch.norm(ball_linvels)))

    # Sensor force penalty (excessive force could indicate instability)
    force_penalty = -torch.norm(sensor_forces, dim=-1)  # Negative reward for too much force
    force_penalty = force_penalty / (1 + torch.exp(-force_penalty_temperature * torch.norm(sensor_forces)))

    # Sensor torque penalty (excessive torque could indicate instability)
    torque_penalty = -torch.norm(sensor_torques, dim=-1)  # Negative reward for too much torque
    torque_penalty = torque_penalty / (1 + torch.exp(-torque_penalty_temperature * torch.norm(sensor_torques)))

    # Total Reward
    total_reward = height_reward + falling_penalty + stationary_penalty + force_penalty + torque_penalty

    # Dictionary of individual components
    reward_components = {
        'height_reward': height_reward,
        'falling_penalty': falling_penalty,
        'stationary_penalty': stationary_penalty,
        'force_penalty': force_penalty,
        'torque_penalty': torque_penalty
    }

    return total_reward, reward_components
