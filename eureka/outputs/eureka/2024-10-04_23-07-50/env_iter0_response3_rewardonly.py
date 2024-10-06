@torch.jit.script
def compute_reward(ball_pos: torch.Tensor, ball_linvels: torch.Tensor, sensor_forces: torch.Tensor, sensor_torques: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants for reward calculation
    FORCE_THRESHOLD = 1.0
    VELOCITY_THRESHOLD = 0.5
    TEMPERATURE_POSITION = 0.1
    TEMPERATURE_FORCE = 0.1
    TEMPERATURE_VELOCITY = 0.1
    
    # Reward components
    reward_position = -torch.exp(-((ball_pos[..., 2] - 1.0) ** 2) / TEMPERATURE_POSITION)  # Encourage ball to stay on the table (z position = 1)
    reward_velocity = -torch.exp(-torch.norm(ball_linvels, dim=-1) / TEMPERATURE_VELOCITY)  # Penalize high linear velocity
    reward_forces = torch.where(torch.norm(sensor_forces, dim=-1) < FORCE_THRESHOLD, torch.tensor(0.0, device=ball_pos.device), -1.0)  # Encourage low sensor forces
    reward_torques = torch.where(torch.norm(sensor_torques, dim=-1) < FORCE_THRESHOLD, torch.tensor(0.0, device=ball_pos.device), -1.0)  # Encourage low sensor torques

    # Total reward
    total_reward = reward_position + reward_velocity + reward_forces + reward_torques

    # Reward components dictionary
    reward_components = {
        "position_reward": reward_position,
        "velocity_reward": reward_velocity,
        "forces_reward": reward_forces,
        "torques_reward": reward_torques,
    }

    return total_reward, reward_components
