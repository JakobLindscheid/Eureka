@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor, 
                   angle_to_target: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity Reward
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    reward_velocity = forward_velocity

    # Force Penalty (changed transformation)
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    force_temperature = 0.05  # Adjusted temperature
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # Stability component (based on angle to target)
    angle_temperature = 1.0
    stability_reward = torch.exp(-angle_temperature * torch.abs(angle_to_target))

    # Joint velocity penalty to encourage smoothness
    dof_vel_temp = 0.1
    penalty_dof_vel = torch.exp(-dof_vel_temp * torch.norm(dof_vel, p=1, dim=1))

    # Total reward
    total_reward = reward_velocity * penalty_force * stability_reward * penalty_dof_vel

    # Reward components dictionary
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force,
        "stability_reward": stability_reward,
        "penalty_dof_vel": penalty_dof_vel
    }

    return total_reward, reward_dict
