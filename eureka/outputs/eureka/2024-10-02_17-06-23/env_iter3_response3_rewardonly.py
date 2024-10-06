@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.5
    alignment_temp = 0.5
    target_reach_temp = 0.2
    movement_penalty_temp = 0.1

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component
    forward_velocity = velocity[:, 0]

    # Normalized speed as a reward component (scale to 0-1)
    max_forward_speed = 5.0  # Assume we know the maximum expected speed
    speed_reward = torch.clamp(forward_velocity / max_forward_speed, min=0.0, max=1.0)
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)

    # Alignment with target direction using dot product with normalized direction
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1) + 1e-6
    direction_to_target_normalized = direction_to_target / distance_to_target.unsqueeze(-1)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / max_forward_speed, min=0.0, max=1.0)
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)

    # New Target Reach Reward based on distance to the target (better feedback)
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)

    # Movement penalty - significantly stronger when the agent does not move
    movement_penalty = -torch.exp(-torch.clamp(torch.norm(velocity, p=2, dim=-1), min=0.01) / movement_penalty_temp)

    # Total reward is the combination of all components
    total_reward = speed_reward_transformed + alignment_reward_transformed + target_reach_reward + movement_penalty

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'alignment_reward': alignment_reward_transformed,
        'target_reach_reward': target_reach_reward,
        'movement_penalty': movement_penalty
    }
    
    return total_reward, reward_components
