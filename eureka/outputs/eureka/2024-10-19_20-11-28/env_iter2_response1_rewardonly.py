@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for forward velocity with normalization
    velocity_temperature = 0.2  # Increased sensitivity
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # Adjust the penalty for joint forces to have a less negative impact
    force_temperature = 0.02  # Further adjusted to lessen its negative impact
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # Combined reward encourages velocity while considering force usage
    total_reward = reward_velocity * penalty_force

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }

    return total_reward, reward_dict
