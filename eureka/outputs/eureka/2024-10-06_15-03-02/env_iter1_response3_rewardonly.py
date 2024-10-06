@torch.jit.script
def compute_reward(velocity: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for reward normalization
    velocity_temp = 0.5
    heading_temp = 0.5

    # Calculate the linear speed of the humanoid
    speed = torch.norm(velocity, p=2, dim=-1)

    # Compute rewards for speed and direction alignment
    speed_reward = torch.exp(velocity_temp * speed)
    
    # The reward for heading alignment (bonus for heading in the intended direction)
    heading_direction_reward = torch.clamp(torch.dot(heading_vec, velocity) / (torch.norm(heading_vec) * torch.norm(velocity) + 1e-6), min=0.0, max=1.0)
    heading_reward = torch.exp(heading_temp * heading_direction_reward)

    # Total reward is a combination of speed reward and heading reward
    total_reward = speed_reward + heading_reward

    # Create a dictionary for individual rewards
    reward_components = {
        'speed_reward': speed_reward,
        'heading_reward': heading_reward,
    }

    return total_reward, reward_components
