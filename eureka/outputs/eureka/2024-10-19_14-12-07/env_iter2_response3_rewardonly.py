@torch.jit.script
def compute_reward(velocity: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature variables for reward transformation
    temp_reward: float = 1.0
    
    # Compute the speed reward
    speed_reward = torch.norm(velocity, p=2, dim=-1) / dt
    
    # Transform the speed reward using an exponential function for normalization
    transformed_speed_reward = torch.exp(temp_reward * speed_reward)

    # Total reward
    total_reward = transformed_speed_reward

    # Create a dictionary for individual reward components
    reward_components = {
        'speed_reward': transformed_speed_reward
    }

    return total_reward, reward_components
