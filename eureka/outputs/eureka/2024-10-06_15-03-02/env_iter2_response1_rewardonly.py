@torch.jit.script
def compute_reward(velocity: torch.Tensor, prev_potentials: torch.Tensor, potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants for reward components
    speed_weight: float = 1.0
    potential_weight: float = 0.5
    
    # Reward for speed (magnitude of velocity vector)
    speed = torch.norm(velocity, p=2, dim=-1)
    speed_reward = speed_weight * speed
    
    # Reward based on change in potential
    potential_change = potentials - prev_potentials
    potential_reward = potential_weight * potential_change
    
    # Total reward
    total_reward = speed_reward + potential_reward
    
    # Create a temperature for normalization (can be tuned)
    temperature_speed = 0.1
    temperature_potential = 0.1
    
    # Normalize rewards using exponential transformation
    normalized_speed_reward = torch.exp(temperature_speed * speed_reward)
    normalized_potential_reward = torch.exp(temperature_potential * potential_reward)
    
    # Total normalized reward
    total_normalized_reward = normalized_speed_reward + normalized_potential_reward

    # Individual reward components
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'potential_reward': normalized_potential_reward
    }

    return total_normalized_reward, reward_components
