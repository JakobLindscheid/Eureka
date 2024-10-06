@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.3
    alignment_temp = 0.3
    target_reach_temp = 0.2
    movement_penalty_temp = 0.1  # Increased strength

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate speed reward (scaled down)
    forward_velocity = velocity[:, 0]
    speed_reward = torch.clamp(forward_velocity / 5.0, min=0.0, max=1.0)  # Stronger scaling

    # Calculate direction to the target
    direction_to_target = targets - torso_position
    direction_to_target_normalized = direction_to_target / (torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6)
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)
    
    # New reward component for moving towards the target
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)  # Exponential decay based on distance

    # Alignment reward
    alignment_reward = torch.sum(velocity * direction_to_target_normalized, dim=-1)
    alignment_reward = torch.clamp(alignment_reward, min=0.0, max=1.0)  # Normalize to 0-1 range
    
    # Strengthening movement penalty for inactivity
    no_movement_penalty = -torch.exp(-torch.norm(velocity, p=2, dim=-1) / movement_penalty_temp)  

    # Aggregate total reward, adjusted to normalize components
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
