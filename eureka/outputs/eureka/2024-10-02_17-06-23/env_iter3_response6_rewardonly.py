@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.1
    alignment_temp = 0.1
    target_reach_temp = 0.5
    movement_penalty_temp = 0.05

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component
    forward_velocity = velocity[:, 0]

    # Speed reward: now encourages higher speed
    speed_reward = torch.clamp(forward_velocity / 3.0, min=0.0, max=1.0)  # Scaling down to allow differentiation
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)

    # New reward component for reaching the target (distance)
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)  # Incentivizes getting closer to targets

    # Alignment with target direction
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / 3.0, min=0.0, max=1.0)  # Scaled to encourage alignment
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)

    # Enhancing movement penalty for remaining stationary
    no_movement_penalty = -torch.exp(-torch.norm(velocity, p=2, dim=-1) / movement_penalty_temp)

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
