@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor, episode_length: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters for reward transformation
    upright_temperature: float = 0.1
    velocity_temperature: float = 0.5  # Increased to penalize high velocities more
    length_temperature: float = 0.01  # Introduced to reward longer episode lengths

    # Extract the positions and velocities of the cart and the pole
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]

    # Reward for keeping the pole upright (closer to 0 radians)
    upright_reward = torch.exp(-upright_temperature * torch.abs(pole_angle))

    # Penalty for large cart velocities (to keep the cart stable)
    velocity_penalty = torch.exp(-velocity_temperature * torch.abs(cart_vel))

    # Reward for longer episode lengths
    length_reward = torch.exp(length_temperature * episode_length)

    # Total reward is a combination of upright reward, velocity penalty, and length reward
    total_reward = upright_reward * velocity_penalty * length_reward

    # Dictionary of individual reward components
    reward_components = {
        "upright_reward": upright_reward,
        "velocity_penalty": velocity_penalty,
        "length_reward": length_reward
    }

    return total_reward, reward_components
