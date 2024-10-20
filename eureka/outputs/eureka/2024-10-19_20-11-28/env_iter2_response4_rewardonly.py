@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor, episode_length: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Emphasize reward for forward velocity, adjusting temperature to scale the improvement more effectively
    velocity_temperature = 0.15
    reward_velocity = forward_velocity * velocity_temperature

    # Adjust force penalty for better prominence
    force_temperature = 0.1  # Increased to further differentiate forces
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # Introduce a bonus for sustaining longer episode lengths
    length_bonus_temperature = 0.02
    reward_length_bonus = torch.exp(episode_length * length_bonus_temperature) - 1.0

    # Total reward combines the components with new weights and an additional bonus for sustained action
    total_reward = reward_velocity * penalty_force + reward_length_bonus

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force,
        "reward_length_bonus": reward_length_bonus
    }

    return total_reward, reward_dict
