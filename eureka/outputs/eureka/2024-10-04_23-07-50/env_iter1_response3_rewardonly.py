@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, table_height: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    temperature_position = 0.1
    temperature_velocity = 0.1
    max_height = table_height
    min_height = 0.0

    # Reward based on the height of the ball
    ball_height = ball_positions[..., 2]  # Assuming z-axis is the height
    reward_height = torch.clamp(ball_height - min_height, min=0.0, max=max_height)

    # Reward based on the ball's vertical velocity
    reward_velocity = -torch.abs(ball_linvels[..., 2])  # Encourage no vertical movement

    # Combine rewards
    total_reward = torch.exp(reward_height / temperature_position) + torch.exp(reward_velocity / temperature_velocity)
    
    # Dictionary of individual reward components
    reward_components = {
        'height_reward': reward_height,
        'velocity_reward': reward_velocity
    }

    return total_reward, reward_components
