@torch.jit.script
def compute_reward(velocity: torch.Tensor, previous_potential: torch.Tensor, potential: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for reward normalization
    temperature_velocity = 0.1
    temperature_potential = 0.1

    # Compute the reward for speed
    speed_reward = torch.norm(velocity, p=2, dim=-1)
    normalized_speed_reward = torch.exp(-temperature_velocity * speed_reward)

    # Compute the reward based on potential change
    potential_change = potential - previous_potential
    potential_reward = torch.clamp(potential_change, min=0)  # Only reward positive changes
    normalized_potential_reward = torch.exp(temperature_potential * potential_reward)

    # Total reward
    total_reward = normalized_speed_reward + normalized_potential_reward

    # Create a dictionary for individual rewards
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'potential_reward': normalized_potential_reward
    }
    
    return total_reward, reward_components
