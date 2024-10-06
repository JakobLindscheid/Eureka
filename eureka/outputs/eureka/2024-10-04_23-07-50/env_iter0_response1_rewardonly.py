@torch.jit.script
def compute_reward(ball_pos: torch.Tensor, ball_linvel: torch.Tensor, table_height: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature for reward transformations
    position_temp = 0.1
    velocity_temp = 0.05
    
    # Calculate reward components
    # 1. Reward based on ball z-position (to keep the ball above the table)
    above_table = ball_pos[..., 2] - table_height
    position_reward = torch.clamp(above_table, min=0.0)  # Reward is positive when ball is at or above the table height
    
    # 2. Reward for keeping the ball stationary (low linear velocity)
    velocity_reward = -torch.norm(ball_linvel, dim=-1)  # Negative reward for higher linear velocities
    
    # Combine rewards
    total_reward = (torch.exp(position_temp * position_reward) + 
                    torch.exp(velocity_temp * velocity_reward))
    
    # Create reward component dictionary
    reward_components = {
        "position_reward": position_reward,
        "velocity_reward": velocity_reward
    }
    
    return total_reward, reward_components
