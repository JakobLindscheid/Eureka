@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extracting forward velocity (assuming target along x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Rescaled reward for forward velocity
    velocity_temperature = 0.3  # Rescaled for more balanced contribution
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # Redefined penalty force for better variable contribution
    force_temperature = 0.1  # Increased temperature for a more noticeable penalty
    force_threshold = 10.0  # Set threshold to define excessive force
    force_magnitude = torch.relu(torch.norm(dof_force_tensor, p=2, dim=1) - force_threshold)
    penalty_force = -torch.exp(force_magnitude * force_temperature)  # Exponential penalty for forces above threshold

    # Weighted reward components with updated balance
    total_reward = reward_velocity + penalty_force * 0.1  # More balanced weighting

    # Create dictionary for reward components' analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }

    return total_reward, reward_dict
