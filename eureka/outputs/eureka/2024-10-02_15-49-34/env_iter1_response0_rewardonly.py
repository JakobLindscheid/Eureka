@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    progress_reward_temp = 0.5
    speed_reward_temp = 0.3
        
    # Calculate the distance to the target
    to_target = target - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)
    
    # Reward for moving closer to the target
    progress_reward = -distance_to_target
    progress_reward_transformed = torch.exp(progress_reward / progress_reward_temp)
    
    # Reward for forward speed (assuming forward direction is along the first axis)
    forward_speed = velocity[:, 0]  # Extracting the forward velocity component
    speed_reward = forward_speed
    speed_reward_transformed = torch.exp(speed_reward / speed_reward_temp)
    
    # Combine rewards
    total_reward = progress_reward_transformed + speed_reward_transformed
    
    reward_components = {
        'progress_reward': progress_reward_transformed,
        'speed_reward': speed_reward_transformed,
    }

    return total_reward, reward_components
