@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    speed_weight = 1.0  # Weight for speed in the reward function
    temperature_speed = 0.05  # Temperature for the speed component

    # Compute speed (magnitude of velocity)
    speed = torch.norm(velocity, p=2, dim=-1)

    # Reward based on speed, normalized
    speed_reward = speed_weight * speed / dt
    normalized_speed_reward = torch.exp(speed_reward / temperature_speed)

    # Total reward is only the speed reward in this case
    total_reward = normalized_speed_reward

    # Individual reward components
    reward_components = {
        'speed_reward': normalized_speed_reward,
    }
    
    return total_reward, reward_components
