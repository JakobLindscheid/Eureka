@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 1.0  # Scale down the temperature to reduce reward impact
    alignment_temp = 1.0
    target_reach_temp = 0.5  # Reduce the distance impact 
    movement_penalty_temp = 0.5  # Decrease penalty strength

    # Extract relevant components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Speed reward based on forward velocity (re-scaled)
    forward_velocity = velocity[:, 0]
    speed_reward = torch.clamp(forward_velocity / 2.0, min=0.0, max=1.0)  # Scaled down

    # Target reach reward based on proximity to target
    distance_to_target = torch.norm(targets - torso_position, p=2, dim=-1)
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)

    # Improved alignment reward based on velocity and normalized target direction
    direction_to_target = targets - torso_position
    direction_to_target_normalized = direction_to_target / (torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1), min=0.0, max=1.0)

    # Adjusting movement penalty for inactivity
    no_movement_penalty = -torch.exp(-torch.norm(velocity, p=2, dim=-1) / movement_penalty_temp)

    # Transform rewards to manage range effectively
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)

    # Total_reward combining all components
    total_reward = speed_reward_transformed + alignment_reward_transformed + target_reach_reward + no_movement_penalty

    # Create a dictionary to hold the individual rewards
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'alignment_reward': alignment_reward_transformed,
        'target_reach_reward': target_reach_reward,
        'movement_penalty': no_movement_penalty
    }
    
    return total_reward, reward_components
