@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Increase sensitivity of velocity reward
    velocity_temperature = 0.2  # Increase temperature to boost velocity impact
    reward_velocity = torch.exp(forward_velocity * velocity_temperature)

    # Adjust penalty for high joint forces to encourage smoother motion
    force_temperature = 0.1  # Adjusted temperature to slightly influence the penalty more
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # Total reward combines velocity reward and force penalty
    total_reward = reward_velocity * penalty_force

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }

    return total_reward, reward_dict
