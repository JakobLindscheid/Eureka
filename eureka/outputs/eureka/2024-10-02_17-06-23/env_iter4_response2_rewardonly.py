@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 1.0
    alignment_temp = 1.0
    target_reach_temp = 0.5
    movement_penalty_temp = 0.3  # Stronger penalty

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component (scaled)
    forward_velocity = velocity[:, 0]  # We consider only the x-direction
    speed_reward = torch.clamp(forward_velocity / 2.0, min=0.0, max=1.0)  # Normalize the maximum expected speed

    # Improved targeting direction and distance calculation
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)

    # Target reach reward based on decreasing distance to target
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)

    # Reformulated alignment reward to encourage movement towards the target
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / 1.5, min=0.0, max=1.0)  # Lower scaling
    
    # Movement penalty: stronger penalty for insufficient movement
    movement_penalty = -torch.exp(-torch.norm(velocity, p=2, dim=-1) / movement_penalty_temp)  

    # Total reward incorporates positive and negative components
    total_reward = (speed_reward + alignment_reward + target_reach_reward + movement_penalty)

    # Create a rewards dictionary
    reward_components = {
        'speed_reward': torch.exp(speed_reward / speed_temp),
        'alignment_reward': torch.exp(alignment_reward / alignment_temp),
        'target_reach_reward': target_reach_reward,
        'movement_penalty': movement_penalty,
    }
    
    return total_reward, reward_components
