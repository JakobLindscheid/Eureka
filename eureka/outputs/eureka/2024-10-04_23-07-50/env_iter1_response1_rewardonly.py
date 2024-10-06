@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, dof_positions: torch.Tensor, dof_velocities: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for reward components
    stability_temp = 1.0
    velocity_temp = 1.0
    position_temp = 1.0

    # Calculate the distance of the ball from the tabletop (assuming y=0 is the tabletop height)
    tabletop_height = 0.0
    ball_height = ball_positions[..., 1]  # Assuming y is the second dimension
    distance_from_tabletop = tabletop_height - ball_height

    # Reward for keeping the ball on the table (positive reward when the ball is above the tabletop)
    stability_reward = torch.exp(stability_temp * distance_from_tabletop).clamp(max=1.0)  # Normalize between 0 and 1

    # Reward for ball's linear velocity; penalize if it moves too fast (higher speed is less desirable)
    speed_penalty = torch.exp(-velocity_temp * torch.norm(ball_linvels, dim=-1)).clamp(max=1.0)

    # Reward for maintaining a stable position of the actuated degrees of freedom (dofs)
    position_reward = torch.exp(position_temp * (1.0 - torch.norm(dof_positions[..., :3], dim=-1))).clamp(max=1.0)

    # Total reward combines all components
    total_reward = stability_reward + speed_penalty + position_reward

    # Prepare individual components for logging or debugging
    reward_components = {
        'stability_reward': stability_reward,
        'speed_penalty': speed_penalty,
        'position_reward': position_reward
    }

    return total_reward, reward_components
