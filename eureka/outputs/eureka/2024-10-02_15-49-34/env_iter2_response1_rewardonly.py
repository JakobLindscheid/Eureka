@torch.jit.script
def compute_reward(torso_position: torch.Tensor, torso_velocity: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.1
    height_temp = 0.1
    
    # Calculate the forward speed (dot product with up vector for projection)
    forward_speed = torso_velocity[:, 0]  # Assuming first index is forward direction
    speed_reward = torch.exp(speed_temp * forward_speed)  # Reward for speed

    # Calculate the height of the torso relative to the target for additional reward
    height_reward = torso_position[:, 2] - targets[:, 2]  # Target height should be adjusted based on the task
    height_reward = torch.exp(height_temp * height_reward)  # Transform height reward
    
    # Combine rewards
    total_reward = speed_reward + height_reward
    
    # Reward dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'height_reward': height_reward
    }
    
    return total_reward, reward_components
