@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.5
    alignment_temp = 0.5
    target_reach_temp = 0.3
    movement_penalty_temp = 0.1

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Reward based on forward speed, normalized
    forward_velocity = velocity[:, 0]
    speed_reward = torch.clamp(forward_velocity / 5.0, min=0.0)
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)

    # Direction to target and calculating alignment
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)

    # New condition for reaching the target
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)

    # Measure alignment based on angle to target
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / 5.0, min=0.0)
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)

    # Movement penalty: stronger penalty for lower than a motion threshold
    low_speed_penalty = (torch.norm(velocity, p=2, dim=-1) < 0.05).float() * -1.0  # Penalty for being too still
    movement_penalty = low_speed_penalty
    movement_penalty_transformed = torch.exp(movement_penalty / movement_penalty_temp)

    # Total reward as a combination of all components with balancing
    total_reward = (
        speed_reward_transformed + 
        alignment_reward_transformed + 
        target_reach_reward + 
        movement_penalty_transformed
    )

    # Prepare individual reward components for return dictionary
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'alignment_reward': alignment_reward_transformed,
        'target_reach_reward': target_reach_reward,
        'movement_penalty': movement_penalty_transformed
    }

    return total_reward, reward_components
