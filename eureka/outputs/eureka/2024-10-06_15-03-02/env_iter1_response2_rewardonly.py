@torch.jit.script
def compute_reward(velocity: torch.Tensor, torso_position: torch.Tensor, target_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the distance to the target
    distance_to_target = torch.norm(target_position - torso_position, p=2, dim=-1)
    
    # Reward for running fast (magnitude of velocity)
    speed_reward = torch.norm(velocity, p=2, dim=-1)
    
    # Reward for getting closer to the target (negative distance)
    distance_reward = -distance_to_target
    
    # Parameters for normalization
    speed_temp = 0.1  # Temperature for speed reward
    distance_temp = 0.1  # Temperature for distance reward
    
    # Transform rewards using temperature parameters
    transformed_speed_reward = torch.exp(speed_temp * speed_reward)
    transformed_distance_reward = torch.exp(distance_temp * distance_reward)
    
    # Total reward
    total_reward = transformed_speed_reward + transformed_distance_reward
    
    # Individual reward components dictionary
    reward_components = {
        "speed_reward": transformed_speed_reward,
        "distance_reward": transformed_distance_reward
    }
    
    return total_reward, reward_components
