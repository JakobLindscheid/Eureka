@torch.jit.script
def compute_reward(velocity: torch.Tensor, prev_potentials: torch.Tensor, potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants for reward shaping
    speed_weight: float = 1.0
    potential_weight: float = 0.1
    
    # Calculate the speed (magnitude of velocity vector)
    speed = torch.norm(velocity, p=2, dim=1)

    # Reward for speed
    speed_reward = speed_weight * speed

    # Potential reward component
    potential_reward = potential_weight * (prev_potentials - potentials)

    # Total reward is the sum of speed reward and potential reward
    total_reward = speed_reward + potential_reward

    # Normalize rewards using a temperature variable
    speed_temp: float = 0.1
    potential_temp: float = 0.05
    
    normalized_speed_reward = torch.exp(total_reward / speed_temp)
    normalized_potential_reward = torch.exp(potential_reward / potential_temp)

    # Return total normalized reward and individual reward components
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'potential_reward': normalized_potential_reward
    }
    
    return total_reward, reward_components
