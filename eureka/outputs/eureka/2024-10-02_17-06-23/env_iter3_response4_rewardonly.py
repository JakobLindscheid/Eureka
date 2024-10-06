@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 1.0
    alignment_temp = 1.0
    target_reach_temp = 0.5
    movement_penalty_temp = 0.1

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Forward velocity component (scale between 0-1)
    forward_velocity = velocity[:, 0]
    speed_reward = torch.clamp(forward_velocity / 2.0, min=0.0, max=1.0)  # Rescaled for normalization
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)

    # Alignment with target direction
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)

    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / 2.0, min=0.0, max=1.0)  # Adjust scale
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)

    # Target reach reward based on how close it is to the target
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)  # Exponential decay as it approaches target

    # Strong penalty for lack of movement: (consider if velocity magnitude is very low)
    movement_penalty = -torch.sum(torch.norm(velocity, p=2, dim=-1) < 0.1).to(velocity.dtype)  # Penalize for staying still
    movement_penalty_transformed = torch.exp(movement_penalty / movement_penalty_temp)

    # Total reward is the combination of all components
    total_reward = speed_reward_transformed + alignment_reward_transformed + target_reach_reward + movement_penalty_transformed

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'alignment_reward': alignment_reward_transformed,
        'target_reach_reward': target_reach_reward,
        'movement_penalty': movement_penalty_transformed
    }

    return total_reward, reward_components
