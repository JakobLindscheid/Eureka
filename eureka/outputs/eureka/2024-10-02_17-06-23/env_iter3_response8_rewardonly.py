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

    # Calculate forward velocity (magnitude)
    forward_velocity = velocity.norm(dim=1)

    # Reward component for moving forward fast (scale to 0-1)
    speed_reward = torch.clamp(forward_velocity / 5.0, min=0.0, max=1.0)
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)

    # Compute the direction to target
    direction_to_target = targets - torso_position
    distance_to_target = direction_to_target.norm(dim=1)
    
    # New reward component for reaching closer to the target
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)

    # Alignment Reward by computing the cosine alignment with respect to the target
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)
    velocity_normalized = velocity / (forward_velocity.unsqueeze(-1) + 1e-6)
    alignment_reward = torch.clamp(torch.sum(direction_to_target_normalized * velocity_normalized, dim=1), min=0.0, max=1.0)
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)

    # Movement penalty for lack of motion
    no_movement_penalty = -torch.exp(-forward_velocity / movement_penalty_temp)

    # Total reward is the sum of all components
    total_reward = speed_reward_transformed + alignment_reward_transformed + target_reach_reward + no_movement_penalty

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'alignment_reward': alignment_reward_transformed,
        'target_reach_reward': target_reach_reward,
        'movement_penalty': no_movement_penalty
    }

    return total_reward, reward_components
