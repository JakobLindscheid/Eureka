@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.1
    alignment_temp = 0.1
    target_reach_temp = 0.5
    movement_penalty_temp = 0.1

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component, we cap it at a high threshold
    forward_velocity = velocity[:, 0]
    speed_reward = torch.clamp(forward_velocity / 5.0, min=0.0, max=1.0)
    speed_reward_transformed = torch.exp(speed_reward / speed_temp) - 1  # Exponential scaled

    # Calculate the direction towards the target
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)

    # New reward component for getting closer to the target
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)  # Decay with distance

    # Alignment with target direction
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / 5.0, min=0.0, max=1.0)
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp) - 1

    # Penalty for not moving to encourage exploration
    no_movement_penalty = -torch.exp(-torch.norm(velocity, p=2, dim=-1) / movement_penalty_temp)  # Negative reward for not moving

    # Total reward is the combination of all components
    total_reward = speed_reward_transformed + alignment_reward_transformed + target_reach_reward + no_movement_penalty

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'alignment_reward': alignment_reward_transformed,
        'target_reach_reward': target_reach_reward,
        'movement_penalty': no_movement_penalty
    }
    
    return total_reward, reward_components
