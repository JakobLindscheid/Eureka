@torch.jit.script
def compute_reward(torso_velocity: torch.Tensor, target_velocity: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters
    speed_temperature = 0.1  # Temperature for speed reward normalization
    
    # Compute the speed by taking the norm of the torso velocity
    speed = torch.norm(torso_velocity, p=2, dim=-1)
    
    # Compute target speed as the norm of the target velocity
    target_speed = torch.norm(target_velocity, p=2, dim=-1)

    # Speed reward component: Encourage running fast
    speed_reward = speed - target_speed

    # Normalize the speed reward with exponential scaling
    exp_speed_reward = torch.exp(speed_temperature * speed_reward)

    # Total reward
    total_reward = exp_speed_reward

    # Create a dictionary for individual components
    reward_components = {
        "speed_reward": exp_speed_reward,
    }
    
    return total_reward, reward_components
