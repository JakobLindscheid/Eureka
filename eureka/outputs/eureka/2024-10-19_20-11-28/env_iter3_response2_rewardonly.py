@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for forward velocity with a higher temperature to emphasize speed
    velocity_temperature = 0.3  # Increased from previous to further incentivize speed
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # Modify penalty for joint forces, making it more discouraging at higher scales
    force_temperature = 0.05  # Increased temperature to impact more
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # Combine rewards to encourage speed with efficient force use
    total_reward = reward_velocity + penalty_force  # Modify combination to reward structure

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }

    return total_reward, reward_dict
