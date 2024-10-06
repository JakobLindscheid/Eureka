@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters for reward shaping
    upright_temperature: float = 0.1
    velocity_temperature: float = 0.1

    # Calculate the uprightness of the pole
    upright_reward = torch.exp(-upright_temperature * torch.abs(dof_pos[:, 1]))

    # Penalize large velocities to stabilize the pole
    velocity_penalty = torch.exp(-velocity_temperature * torch.abs(dof_vel[:, 1]))

    # Total reward is a combination of uprightness and velocity penalty
    total_reward = upright_reward * velocity_penalty

    # Return the total reward and individual components
    reward_components = {
        "upright_reward": upright_reward,
        "velocity_penalty": velocity_penalty
    }

    return total_reward, reward_components
