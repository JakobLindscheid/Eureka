@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, dof_positions: torch.Tensor, dof_velocities: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for reward components
    stability_temp = 0.5  # Lower temperature for stability reward for sensitivity
    speed_temp = 2.0      # Higher penalty scaling for speed penalty
    position_temp = 0.5   # Lowered to allow more variability in position assessment

    # Calculate the distance of the ball from the tabletop (assuming y=0 is the tabletop height)
    tabletop_height = 0.0
    ball_height = ball_positions[..., 1]  # Assuming y is the second dimension
    distance_from_tabletop = tabletop_height - ball_height

    # Reward for keeping the ball on top of the tabletop (positive reward)
    stability_reward = torch.exp(stability_temp * distance_from_tabletop).clamp(max=1.0)

    # Implement speed penalty; stronger negative influence for higher speeds
    speed_penalty = -torch.exp(speed_temp * torch.norm(ball_linvels, dim=-1)).clamp(min=0.0)

    # Reward for maintaining a stable position of the actuated degrees of freedom
    # Introduce penalties for deviations from ideal position values
    ideal_position = torch.zeros_like(dof_positions[..., :3])  # Assuming ideal position is zero for simplification 
    position_deviation = torch.norm(dof_positions[..., :3] - ideal_position, dim=-1)
    position_reward = torch.exp(position_temp * (1.0 - position_deviation)).clamp(max=1.0)

    # Total reward combines all components
    total_reward = stability_reward + speed_penalty + position_reward

    # Prepare individual components for logging or debugging
    reward_components = {
        'stability_reward': stability_reward,
        'speed_penalty': speed_penalty,
        'position_reward': position_reward
    }

    return total_reward, reward_components
