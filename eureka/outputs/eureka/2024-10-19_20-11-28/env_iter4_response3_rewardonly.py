@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Adjusted reward for forward velocity, scaling down to moderate the impact of large values
    scaled_velocity = forward_velocity / 10.0
    velocity_temperature = 0.3  # Reduced to moderate the effect of speed on the reward
    reward_velocity = torch.exp(scaled_velocity * velocity_temperature) - 1.0

    # Introduced meaningful scaling for force penalty; updated reward function
    force_temperature = 0.01  # New, lower temperature to enhance sensitivity to changes in force
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = -torch.clamp(force_temperature * force_magnitude, max=1.0)  # Ensuring penalty remains within [-1, 0]

    # Balanced combination of reward components, balancing between velocity and efficiency
    total_reward = reward_velocity + penalty_force  # Balanced to ensure both components are contributing effectively

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }

    return total_reward, reward_dict
