@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity (assuming target run direction is x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for high forward velocity
    velocity_temperature = 0.2   # Adjusted to increase velocity sensitivity
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) 

    # Penalize high joint forces
    force_temperature = 0.1  # Increasing sensitivity to penalize inefficient force usage
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # Bonus to reward long survival and sustained runs
    survival_bonus = 0.05

    # Total reward includes velocity and efficient force usage
    total_reward = reward_velocity * penalty_force + survival_bonus

    # Create a reward dictionary for individual analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force,
        "survival_bonus": torch.full_like(reward_velocity, survival_bonus)  # To match batch size
    }

    return total_reward, reward_dict
