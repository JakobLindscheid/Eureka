@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]
    forward_temp = 0.5  # Adjusted temperature for aggressive scaling
    forward_reward = torch.exp(forward_speed / forward_temp) - 1  # Exponential scaling

    # Distance reward: reward for closeness to the target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance
    max_distance = 10.0  # Maximum distance for scaling
    proximity_temp = 1.0  # Temperature for proximity reward scaling
    distance_reward = torch.exp((max_distance - distance_to_target) / proximity_temp) - 1

    # New task motivation reward: positive contributions for reaching the target
    progress_reward = torch.where(distance_to_target < max_distance, 
                                  max_distance - distance_to_target, 
                                  torch.tensor(0.0, device=root_states.device))
    
    # Total reward
    total_reward = forward_reward + distance_reward + progress_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'distance_reward': distance_reward,
        'progress_reward': progress_reward
    }

    return total_reward, reward_components
