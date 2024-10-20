@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Adjusted reward for forward velocity: smoother scaling with a moderate temperature to handle fluctuations
    velocity_temperature = 0.3  # Optimized to balance range and incentive effect
    reward_velocity = torch.sigmoid(forward_velocity * velocity_temperature)

    # Revised force penalty: Reducing magnitude and boosting effectiveness
    force_temperature = 0.1  # Increased to improve dynamic response
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = -torch.sigmoid(force_temperature * force_magnitude)  # Modify to a torque dynamic penalty

    # Combine the reward components, ensuring they contribute cohesively
    total_reward = reward_velocity + penalty_force  # Simplified combination enhancing learning signal coherence

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }

    return total_reward, reward_dict
