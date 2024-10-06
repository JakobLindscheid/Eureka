@torch.jit.script
def compute_reward(torso_position: torch.Tensor, torso_velocity: torch.Tensor, target_position: torch.Tensor, tilt_angle: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Normalization temperature parameters
    speed_temp = 0.1
    distance_temp = 0.1
    
    # Calculate forward direction (assuming forward is along the x-axis)
    forward_direction = torch.tensor([1.0, 0.0, 0.0], device=torso_position.device)
    
    # Reward for speed (magnitude of forward velocity)
    speed_reward = torch.dot(torso_velocity, forward_direction)
    
    # Reward for achieving target position (penalizing distance from target)
    distance_reward = -torch.norm(torso_position - target_position, p=2)
    
    # Combine rewards with temperature normalization
    total_reward = torch.exp(speed_temp * speed_reward) + torch.exp(distance_temp * distance_reward)
    
    # Create a dictionary for individual reward components
    reward_components = {
        'speed_reward': speed_reward,
        'distance_reward': distance_reward,
    }
    
    return total_reward, reward_components
