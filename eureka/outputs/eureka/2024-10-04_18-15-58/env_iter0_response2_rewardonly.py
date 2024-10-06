@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters for reward scaling
    position_temperature: float = 0.1
    velocity_temperature: float = 0.1

    # Extract the relevant DOF positions and velocities
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_vel = dof_vel[:, 0]
    pole_angular_vel = dof_vel[:, 1]

    # Reward for keeping the pole upright (angle close to zero)
    angle_reward = torch.exp(-position_temperature * pole_angle**2)

    # Penalty for large angular velocity of the pole
    angular_vel_penalty = torch.exp(-velocity_temperature * pole_angular_vel**2)

    # Combine rewards
    total_reward = angle_reward * angular_vel_penalty

    # Dictionary of individual reward components
    reward_components = {
        "angle_reward": angle_reward,
        "angular_vel_penalty": angular_vel_penalty
    }

    return total_reward, reward_components
