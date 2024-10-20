@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Modified reward for forward velocity with adjusted normalization
    velocity_temperature = 0.5  # Increased to add more weight to speed
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # Modified force penalty to provide a stronger regularization signal
    force_temperature = 0.05  # Increased to make the penalty more significant
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude) - 1.0  # Adjusted to provide a more negative penalty

    # Combine the reward components, emphasizing velocity more
    total_reward = reward_velocity + 0.5 * penalty_force  # Adjusted weighting to balance velocity and force

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }

    return total_reward, reward_dict
