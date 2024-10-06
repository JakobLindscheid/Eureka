@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, dof_positions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for reward components
    stability_temp = 2.0  # Increased sensitivity for stability
    velocity_temp = 0.5   # Stronger penalty for high speed
    position_temp = 1.0   # Keeping the normalization for position reward

    # Calculate the height of the ball from the tabletop
    tabletop_height = 0.0
    ball_height = ball_positions[..., 1]  # Assuming y is the second dimension
    distance_from_tabletop = tabletop_height - ball_height

    # Reward for keeping the ball on the table (positive reward when the ball is above the tabletop)
    stability_reward = torch.exp(stability_temp * distance_from_tabletop).clamp(max=1.0)

    # Enhanced penalty for ball's linear velocity; penalize if it moves too fast
    speed_penalty = torch.exp(-velocity_temp * torch.norm(ball_linvels, dim=-1)).clamp(max=1.0)

    # Create dynamic position reward; not just a norm, but also a penalty for deviation from desired position
    position_reward = torch.exp(position_temp * (1.0 - torch.clamp(torch.norm(dof_positions[..., :3] - 0.0, dim=-1), max=1)))  # Assuming zero is the desired position

    # Total reward combines all components
    total_reward = stability_reward + speed_penalty + position_reward

    # Prepare individual components for logging or debugging
    reward_components = {
        'stability_reward': stability_reward,
        'speed_penalty': speed_penalty,
        'position_reward': position_reward
    }

    return total_reward, reward_components
