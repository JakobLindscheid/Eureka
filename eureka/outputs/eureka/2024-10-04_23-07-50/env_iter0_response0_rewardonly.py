@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_linvels: torch.Tensor, sensor_forces: torch.Tensor, temperature_position: float = 0.1, temperature_velocity: float = 0.1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Initialize reward components
    position_reward = torch.zeros(ball_positions.size(0), device=ball_positions.device)
    velocity_reward = torch.zeros(ball_linvels.size(0), device=ball_linvels.device)

    # Reward for keeping the ball on the table
    is_ball_above_table = (ball_positions[..., 2] > 0).float()  # Positive reward if ball is above the table (assumed z > 0)
    position_reward += is_ball_above_table

    # Negative reward for falling (if the ball's z position drops below 0)
    position_reward -= (1 - is_ball_above_table)  # Penalize if the ball is below the table

    # Reward for keeping the ball stationary (experience good stability)
    velocity_penalty = torch.clamp(torch.norm(ball_linvels, dim=-1), max=1.0)  # Limit the penalty to avoid extreme values
    velocity_reward -= velocity_penalty  # Penalize for non-zero ball velocity

    # Combine rewards, normalized with temperature parameters
    total_reward = torch.exp(temperature_position * position_reward) + torch.exp(temperature_velocity * velocity_reward)

    # Create the reward components dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward
    }
    
    return total_reward, reward_components
