@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_scale = 40.0  # Higher scaling factor for better incentivization
    velocity_scale = -1.5    # More aggressive penalty for higher velocities
    task_score_scale = 2.0    # Increased reward for successful hovering
    position_temp = 0.2
    velocity_temp = 0.1
    task_score_temp = 1.0

    # Calculate position error
    position_error = root_positions - target_position
    position_distance_sq = torch.norm(position_error, dim=-1) ** 2  # Use squared distance

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_scale * position_distance_sq)  # Encourage getting closer
    velocity_reward = torch.exp(velocity_temp * velocity_scale * velocity_magnitude)  # Encourage low velocities

    # New task score that rewards proximity to target
    proximity_threshold = 0.3  # A defined distance for successful hovering
    task_score = torch.where(position_distance_sq < proximity_threshold ** 2, 
                              torch.tensor(1.0, device=root_positions.device), 
                              torch.tensor(0.0, device=root_positions.device))
    task_score_reward = task_score * task_score_scale  # Scale up the task score

    # Total reward
    total_reward = position_reward + velocity_reward + task_score_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
        'task_score_reward': task_score_reward,
    }

    return total_reward, reward_components
