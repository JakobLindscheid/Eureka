@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, table_height: float, ball_radius: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward parameters
    height_temperature = 0.1
    stable_temperature = 0.05

    # Compute rewards
    ball_height = ball_positions[..., 2]  # Assuming z-axis is the height
    reward_height = torch.clamp(table_height - (ball_height + ball_radius), min=0.0)  # Reward for being above the table
    reward_stability = torch.exp(-torch.std(ball_positions[..., 0:2], dim=-1) / stable_temperature)  # Reward based on stability (x and y position variance)

    # Combine rewards
    total_reward = reward_height + reward_stability

    # Prepare reward components for output
    reward_components = {
        'reward_height': reward_height,
        'reward_stability': reward_stability,
    }

    return total_reward, reward_components
