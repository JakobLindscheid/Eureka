@torch.jit.script
def compute_reward(velocity: torch.Tensor, target_velocity: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward based on the speed of the humanoid relative to the target velocity
    speed_reward = torch.norm(velocity, p=2, dim=-1) - target_velocity
    
    # Normalizing the speed reward with a temperature parameter
    temperature_speed = 1.0  # constant temperature for speed reward normalization
    normalized_speed_reward = torch.exp(speed_reward / temperature_speed)

    # Deducting a penalty for time taken to encourage faster running
    time_penalty = -dt
    
    # Total reward combining speed reward and time penalty
    total_reward = normalized_speed_reward + time_penalty

    # Create reward components dictionary
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'time_penalty': time_penalty
    }
    
    return total_reward, reward_components
