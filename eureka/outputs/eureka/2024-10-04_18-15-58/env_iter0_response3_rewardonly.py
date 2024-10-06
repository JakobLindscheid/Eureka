@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters
    pole_angle_threshold = torch.tensor(0.2, dtype=torch.float32)
    cart_position_threshold = torch.tensor(2.4, dtype=torch.float32)
    velocity_penalty_temperature = torch.tensor(0.1, dtype=torch.float32)
    angle_penalty_temperature = torch.tensor(1.0, dtype=torch.float32)
    position_penalty_temperature = torch.tensor(0.5, dtype=torch.float32)

    # Extract relevant states
    cart_position = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_velocity = dof_vel[:, 0]
    pole_angular_velocity = dof_vel[:, 1]

    # Reward components
    angle_penalty = torch.exp(-angle_penalty_temperature * torch.abs(pole_angle))
    position_penalty = torch.exp(-position_penalty_temperature * torch.abs(cart_position))
    velocity_penalty = torch.exp(-velocity_penalty_temperature * torch.abs(cart_velocity + pole_angular_velocity))

    # Total reward
    reward = angle_penalty * position_penalty * velocity_penalty

    # Individual reward components dictionary
    reward_components = {
        "angle_penalty": angle_penalty,
        "position_penalty": position_penalty,
        "velocity_penalty": velocity_penalty
    }

    return reward, reward_components
