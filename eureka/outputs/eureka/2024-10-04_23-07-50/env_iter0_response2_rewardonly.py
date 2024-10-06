@torch.jit.script
def compute_reward(ball_positions: torch.Tensor, ball_velocities: torch.Tensor, tabletop_height: float,
                   temperature_position: float, temperature_velocity: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward components
    position_reward = torch.exp(-(ball_positions[..., 2] - tabletop_height) ** 2 / (2.0 ** 2))  # Encourage staying on the table
    velocity_reward = torch.exp(-torch.norm(ball_velocities, dim=-1) / 2.0)  # Encourage low velocity

    # Total reward is a combination of position and velocity rewards
    total_reward = position_reward + velocity_reward
    
    # Individual reward components
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
    }
    
    return total_reward, reward_components
