@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor, pole_angle_threshold: float = 0.2) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the relevant DOF positions and velocities
    cart_pos = dof_pos[:, 0]
    cart_vel = dof_vel[:, 0]
    pole_angle = dof_pos[:, 1]
    pole_vel = dof_vel[:, 1]

    # Reward for keeping the pole upright (close to zero angle)
    upright_reward_temp = 10.0  # Temperature parameter for upright reward
    upright_reward = torch.exp(-upright_reward_temp * torch.abs(pole_angle))

    # Penalty for the cart moving too far from the center
    cart_pos_penalty_temp = 0.1  # Temperature parameter for cart position penalty
    cart_pos_penalty = torch.exp(-cart_pos_penalty_temp * torch.abs(cart_pos))

    # Penalty for the pole moving too fast
    pole_vel_penalty_temp = 0.05  # Temperature parameter for pole velocity penalty
    pole_vel_penalty = torch.exp(-pole_vel_penalty_temp * torch.abs(pole_vel))

    # Total reward
    total_reward = upright_reward * cart_pos_penalty * pole_vel_penalty

    # Dictionary of individual reward components
    reward_components = {
        "upright_reward": upright_reward,
        "cart_pos_penalty": cart_pos_penalty,
        "pole_vel_penalty": pole_vel_penalty
    }

    return total_reward, reward_components
