@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.5
    alignment_temp = 0.5
    target_reach_temp = 1.0
    movement_penalty_temp = 0.1

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Forward velocity component (cap to [0,1])
    forward_velocity = velocity[:, 0]
    speed_reward = torch.clamp(forward_velocity / 2.5, min=0.0, max=1.0)  # Normalize speed to [0, 1]

    # Calculate the direction towards the target for the alignment reward
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)
    
    # New alignment reward: Encourages agent to move towards the target
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / 2.0, min=0.0, max=1.0)
    
    # New target reach reward: Inverse distance to target with normalization
    target_reach_reward = torch.clamp(1.0 - (distance_to_target / 10.0), min=0.0, max=1.0)  # Normalize over a range
    
    # Increased movement penalty for lack of movement
    no_movement_penalty = -torch.mean(torch.exp(-torch.norm(velocity, p=2, dim=-1) / movement_penalty_temp))
    
    # Total reward combines all components after transformations
    total_reward = (torch.exp(speed_reward / speed_temp) +
                    torch.exp(alignment_reward / alignment_temp) +
                    target_reach_reward +
                    no_movement_penalty)

    # Create a rewards dictionary
    reward_components = {
        'speed_reward': torch.exp(speed_reward / speed_temp),
        'alignment_reward': torch.exp(alignment_reward / alignment_temp),
        'target_reach_reward': target_reach_reward,
        'movement_penalty': no_movement_penalty
    }
    
    return total_reward, reward_components
