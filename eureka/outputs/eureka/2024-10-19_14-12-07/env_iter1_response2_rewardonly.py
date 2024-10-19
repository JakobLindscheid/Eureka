@torch.jit.script
def compute_reward(velocity: torch.Tensor, target_speed: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature for normalization
    temperature = 1.0
    
    # Compute the speed of the humanoid
    speed = torch.norm(velocity, p=2, dim=-1)
    
    # Reward based on how close the humanoid's speed is to the target speed
    speed_reward = speed / target_speed  # Normalize by target speed
    
    # Transform the reward with temperature
    transformed_speed_reward = torch.exp(speed_reward / temperature)
    
    # Total reward is the transformed speed reward
    total_reward = transformed_speed_reward
    
    # Create a dictionary of individual rewards
    individual_rewards = {
        "speed_reward": transformed_speed_reward
    }
    
    return total_reward, individual_rewards
