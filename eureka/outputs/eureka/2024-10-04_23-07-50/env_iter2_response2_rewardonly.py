@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, dof_positions: torch.Tensor, dof_velocities: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for reward components
    stability_temp = 1.0
    speed_temp = 2.0  # Increase the sensitivity of the speed penalty
    position_temp = 0.5  # Decrease the impact of position reward

    # Calculate the distance of the ball from the tabletop (assuming y=0 is the tabletop height)
    tabletop_height = 0.0
    ball_height = ball_positions[..., 1]  # Assuming y is the second dimension
    distance_from_tabletop = tabletop_height - ball_height

    # Reward for keeping the ball on the table (positive reward when the ball is above the tabletop)
    stability_reward = torch.exp(stability_temp * distance_from_tabletop).clamp(max=1.0)  # Normalize between 0 and 1

    # Penalize speed more aggressively by squaring the norm of the ball's linear velocity
    speed_penalty = torch.exp(-speed_temp * (torch.norm(ball_linvels, dim=-1) ** 2)).clamp(max=1.0)

    # Reward for maintaining a stable position of the actuated degrees of freedom (dofs) with reduced influence
    position_reward = torch.exp(position_temp * (1.0 - torch.norm(dof_positions[..., :3], dim=-1))).clamp(max=0.5)  # Scale down the max value

    # Total reward combines all components; give more weight to stability and speed
    total_reward = stability_reward * 0.6 + speed_penalty * 0.3 + position_reward * 0.1

    # Prepare individual components for logging or debugging
    reward_components = {
        'stability_reward': stability_reward,
        'speed_penalty': speed_penalty,
        'position_reward': position_reward
    }

    return total_reward, reward_components
