@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 2.0
    alignment_temp = 2.0
    target_reach_temp = 1.0

    # Extract positions
    torso_position = root_states[:, 0:3]
    
    # Forward velocity component normalized
    forward_velocity = velocity[:, 0]
    speed_reward = torch.clamp(forward_velocity / 3.0, min=0.0, max=1.0)  # Scale down speed reward
    
    # Calculate alignment with target
    direction_to_target = targets - torso_position
    norm_direction = torch.norm(direction_to_target, p=2, dim=-1)
    alignment_reward = torch.clamp(torch.cos(torch.atan2(direction_to_target[:, 1], direction_to_target[:, 0])), min=0.0, max=1.0)

    # Target reaching reward based on distance
    distance_reward = torch.exp(-norm_direction / target_reach_temp)

    # Transform rewards to ensure they have an effective impact
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)
    
    # Total reward combines all beneficial components
    total_reward = speed_reward_transformed + alignment_reward_transformed + distance_reward

    # Create a rewards dictionary
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'alignment_reward': alignment_reward_transformed,
        'distance_reward': distance_reward
    }
    
    return total_reward, reward_components
