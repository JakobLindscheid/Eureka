@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.5
    alignment_temp = 1.0
    target_reach_temp = 0.5
    movement_penalty_temp = 0.5

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component (scaled down)
    forward_velocity = velocity[:, 0] * 0.5  # Scale down the impact of speed
    speed_reward = torch.clamp(forward_velocity / 3.0, min=0.0, max=1.0) 
    
    # Calculate the direction towards the target (normalize to unit vector)
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)

    # Redefine target reach reward as a function of distance to target
    target_reach_reward = -torch.exp(-distance_to_target / target_reach_temp)

    # Improved alignment reward to guide the agent to move in the direction of the target
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1), min=0.0, max=1.0)
    
    # Transforming rewards to ensure they have an effective impact
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)
    
    # Movement penalty that encourages continuous movement
    no_movement_penalty = -torch.exp(-torch.norm(velocity, p=2, dim=-1) / movement_penalty_temp)  

    # Total reward combines all beneficial components
    total_reward = speed_reward_transformed + alignment_reward_transformed + target_reach_reward + no_movement_penalty

    # Create a rewards dictionary
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'alignment_reward': alignment_reward_transformed,
        'target_reach_reward': target_reach_reward,
        'movement_penalty': no_movement_penalty
    }
    
    return total_reward, reward_components
