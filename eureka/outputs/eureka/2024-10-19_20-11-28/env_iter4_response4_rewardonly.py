@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Modified reward for forward velocity, with scaling
    velocity_temperature = 0.3  # Reduced to moderate the scale
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # Re-write the force penalty with a larger force threshold consideration
    max_force_thr = 200.0  # Example threshold for normalizing
    force_temperature = 0.01  # Fine-tuned temperature parameter
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1) / max_force_thr
    penalty_force = torch.clamp(1.0 - force_temperature * force_magnitude, min=-1.0, max=0.0)

    # Combine the reward components
    total_reward = reward_velocity + penalty_force

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }

    return total_reward, reward_dict
