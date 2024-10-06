@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, tabletop_height: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    reward_for_staying_on_table = 1.0
    reward_for_velocity = 0.1
    penalty_for_falling = -10.0
    
    # Checks if the ball is above the tabletop
    on_table = ball_positions[..., 2] >= tabletop_height
    
    # Reward components
    reward_staying_on_table = torch.where(on_table, torch.tensor(reward_for_staying_on_table, device=ball_positions.device), torch.tensor(penalty_for_falling, device=ball_positions.device))
    
    # Velocity reward, penalizing high speeds
    speed_penalty = -torch.sum(ball_linvels ** 2, dim=-1) * 0.01  # Small penalty for high velocities
    
    # Total reward
    total_reward = reward_staying_on_table + speed_penalty
    
    # Detailed reward dictionary
    reward_components = {
        'stay_on_table': reward_staying_on_table,
        'velocity_penalty': speed_penalty
    }

    return total_reward, reward_components
