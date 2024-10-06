@torch.jit.script
def compute_reward(root_positions: torch.Tensor, 
                   target_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature variable for transformed rewards
    distance_temp = 0.1  # Temperature for distance penalty
    hover_temp = 0.05    # Temperature for hover stability
    
    # Calculate the distances to the target position
    distance_to_target = torch.norm(root_positions[:, :3] - target_position, dim=1)
    
    # Reward based on distance: closer -> higher reward
    distance_reward = -distance_to_target  # We want to minimize distance

    # Penalty for not hovering at the target height
    hover_penalty = torch.abs(root_positions[:, 2] - target_position[2])
    
    # Total reward
    total_reward = torch.exp(distance_temp * distance_reward) - torch.exp(hover_temp * hover_penalty)

    # Construct individual reward components dictionary
    individual_rewards = {
        'distance_reward': distance_reward,
        'hover_penalty': -hover_penalty
    }
    
    return total_reward, individual_rewards
