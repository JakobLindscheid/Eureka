@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.2
    alignment_temp = 0.2
    target_reach_temp = 0.5
    movement_penalty_temp = 0.1

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component
    forward_velocity = velocity[:, 0]

    # Reward component for moving forward quickly (we expect 2.0 for optimal speed)
    speed_reward = torch.clamp(forward_velocity / 2.0, min=0.0, max=1.0)
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)

    # Calculate the direction towards the target
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)

    # New reward component for getting closer to the target (scale using a distance-based formulation)
    target_reach_reward = torch.clamp(1.0 - distance_to_target / 10.0, min=0.0, max=1.0)
    target_reach_reward_transformed = torch.exp(target_reach_reward / target_reach_temp)

    # Calculate alignment with target direction
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / (torch.norm(velocity, p=2, dim=-1) + 1e-6), min=0.0, max=1.0)
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)

    # Movement penalty (increased sensitivity for remaining stationary)
    no_movement_penalty = -torch.clamp(torch.sum(torch.abs(velocity), dim=1) / 5.0, min=0.0, max=1.0)
    no_movement_penalty_transformed = torch.exp(no_movement_penalty / movement_penalty_temp)

    # Total reward is the combination of all components
    total_reward = speed_reward_transformed + alignment_reward_transformed + target_reach_reward_transformed + no_movement_penalty_transformed

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'alignment_reward': alignment_reward_transformed,
        'target_reach_reward': target_reach_reward_transformed,
        'movement_penalty': no_movement_penalty_transformed
    }
    
    return total_reward, reward_components
