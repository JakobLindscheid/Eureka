@torch.jit.script
def compute_reward(torso_velocity: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature for normalization
    temp_velocity = 1.0
    
    # Compute speed as the norm of the velocity vector
    speed = torch.norm(torso_velocity, p=2, dim=-1)
    
    # Calculate reward based on speed normalized by timestep
    reward_speed = speed / dt  # Reward for speed
    reward = torch.exp(reward_speed / temp_velocity)  # Normalizing the reward
    
    # Collect individual rewards components
    reward_components = {
        "speed_reward": reward_speed,
    }
    
    return reward, reward_components
