@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for forward velocity with normalization
    velocity_temperature = 0.1
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # Revise penalization of high joint forces to encourage smoother motion
    force_temperature = 0.05  # Adjusted temperature parameter to increase sensitivity
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
