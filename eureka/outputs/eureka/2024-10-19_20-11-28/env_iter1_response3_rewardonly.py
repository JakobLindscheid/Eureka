@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity (assuming x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Reward for forward velocity
    reward_velocity_temperature = 0.1
    reward_velocity = torch.exp(reward_velocity_temperature * forward_velocity)
    
    # Penalize high joint forces
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force_temperature = 5.0
    penalty_force = torch.exp(-penalty_force_temperature * force_magnitude)
    
    # Bonus for maintaining stable velocities (low changes in potential)
    delta_potential = torch.abs(potentials - prev_potentials)
    stability_bonus_temperature = 10.0
    stability_bonus = torch.exp(-stability_bonus_temperature * delta_potential)
    
    # Total reward combining different components
    total_reward = reward_velocity * penalty_force * stability_bonus

    # Reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force,
        "stability_bonus": stability_bonus,
    }

    return total_reward, reward_dict
