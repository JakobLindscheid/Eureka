@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor, episode_length: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for forward velocity with enhanced temperature scaling
    velocity_temperature = 0.3  # Adjusted for stronger velocity encouragement
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # Penalty for excessive joint forces with a more punitive scale 
    force_temperature = 0.05  # Increased to make penalty significant
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # Reward for longer episode lengths
    length_temperature = 0.01  # Small reward rate per time unit
    reward_length = episode_length * length_temperature

    # Total reward combines the components, emphasizing running efficiency and lengthy episodes
    total_reward = reward_velocity * penalty_force + reward_length

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force,
        "reward_length": reward_length
    }

    return total_reward, reward_dict
