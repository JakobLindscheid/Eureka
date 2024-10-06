@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters for reward transformation
    upright_temperature: float = 0.1
    cart_velocity_temperature: float = 0.05
    pole_velocity_temperature: float = 0.05

    # Extract the positions and velocities of the cart and the pole
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]

    # Reward for keeping the pole upright (closer to 0 radians)
    upright_reward = torch.exp(-upright_temperature * torch.abs(pole_angle))

    # Penalty for large cart velocities (to keep the cart stable)
    cart_velocity_penalty = torch.exp(-cart_velocity_temperature * torch.abs(cart_vel))

    # Penalty for large pole velocities (to keep the pole stable)
    pole_velocity_penalty = torch.exp(-pole_velocity_temperature * torch.abs(pole_vel))

    # Total reward is a combination of upright reward and velocity penalties
    total_reward = upright_reward * cart_velocity_penalty * pole_velocity_penalty

    # Dictionary of individual reward components
    reward_components = {
        "upright_reward": upright_reward,
        "cart_velocity_penalty": cart_velocity_penalty,
        "pole_velocity_penalty": pole_velocity_penalty
    }

    return total_reward, reward_components
