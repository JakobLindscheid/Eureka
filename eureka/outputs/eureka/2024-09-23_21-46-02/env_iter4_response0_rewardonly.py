@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_scale = 10.0  # Scale for position; more sensitive
    velocity_target = 0.0    # Target velocity to achieve
    velocity_scale = 0.5      # Reward for being within target velocity range
    task_score_scale = 5.0    # Increased positivity for task scoring
    position_temp = 0.5
    velocity_temp = 1.0
    task_score_temp = 0.2

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * (torch.clamp(position_distance, min=0.0, max=1.0) ** 2) / position_scale)  # Aggressive reward closer to target
    velocity_reward = torch.exp(-velocity_temp * torch.clamp(velocity_magnitude, max=1.0))  # Reward for being close to zero velocity
    
    # New task score based on proximity to the target position
    task_score = torch.where(position_distance < 0.2, torch.tensor(1.0, device=root_positions.device), torch.tensor(0.0, device=root_positions.device))
    task_score_reward = task_score * task_score_scale  # Encourage hovering sufficiently near target

    # Total reward
    total_reward = position_reward + velocity_reward + task_score_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
        'task_score_reward': task_score_reward,
    }

    return total_reward, reward_components
