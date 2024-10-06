@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, dof_positions: torch.Tensor, dof_velocities: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for reward components
    stability_temp = 0.5  # Increased sensitivity to distance from tabletop
    speed_temp = 2.0      # More significant penalty for speed
    position_temp = 0.5   # Balance for position reward

    # Calculate the vertical distance of the ball relative to the tabletop
    tabletop_height = 0.0
    ball_height = ball_positions[..., 1]  # Assuming y is the second dimension
    distance_from_tabletop = tabletop_height - ball_height

    # Reformulated stability reward with added penalties if the ball falls too low
    stability_reward = torch.exp(stability_temp * torch.clamp(distance_from_tabletop, min=-1.0, max=0.0))  # Less reward for falling
    
    # Amplified speed penalty
    speed_penalty = torch.exp(-speed_temp * torch.norm(ball_linvels, dim=-1)).clamp(max=1.0)
    
    # Reformulated position reward focusing on deviation from desired positions
    ideal_dof_positions = torch.zeros_like(dof_positions[..., :3])  # Assuming desired position is zero
    position_diff = torch.norm(dof_positions[..., :3] - ideal_dof_positions, dim=-1)
    position_reward = torch.exp(position_temp * (1.0 - position_diff)).clamp(max=1.0)  # Focus on minimizing this distance
    
    # Total reward combines all components
    total_reward = stability_reward + speed_penalty + position_reward

    # Prepare individual components for logging or debugging
    reward_components = {
        'stability_reward': stability_reward,
        'speed_penalty': speed_penalty,
        'position_reward': position_reward
    }

    return total_reward, reward_components
