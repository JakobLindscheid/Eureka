@torch.jit.script
def compute_reward(velocity: torch.Tensor, max_speed: float, prev_potentials: torch.Tensor, current_potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature for reward transformation
    temp_speed = 0.1  # temperature for speed reward
    
    # Reward based on forward velocity (consider only the x direction for running speed)
    forward_velocity = velocity[:, 0]
    speed_reward = torch.mean(forward_velocity) / max_speed  # Normalize by max speed
    speed_reward = torch.exp(temp_speed * speed_reward) - 1  # Transform using exponential
    
    # Reward based on potential changes (negative if less potential)
    potential_change = current_potentials - prev_potentials
    potential_reward = torch.mean(potential_change)
    
    # Total reward calculation
    total_reward = speed_reward + potential_reward
    
    # Creating a dictionary of reward components
    reward_components = {
        "speed_reward": speed_reward,
        "potential_reward": potential_reward
    }
    
    return total_reward, reward_components
