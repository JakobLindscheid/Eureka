@torch.jit.script
def compute_reward(velocity: torch.Tensor, target_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature for reward normalization
    velocity_temp = 0.1

    # Reward for speed: closer to the target velocity is better
    speed_reward = -torch.norm(velocity - target_velocity, p=2)

    # Applying exponential transformation to normalize
    transformed_speed_reward = torch.exp(speed_reward / velocity_temp)

    # Total reward is the transformed speed reward
    total_reward = transformed_speed_reward

    return total_reward, {'speed_reward': transformed_speed_reward}
