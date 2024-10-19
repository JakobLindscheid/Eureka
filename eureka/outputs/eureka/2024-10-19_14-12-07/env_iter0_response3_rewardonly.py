@torch.jit.script
def compute_reward(velocity: torch.Tensor, target_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    speed_weight = 1.0  # Weight for speed reward
    direction_weight = 0.5  # Weight for direction alignment

    # Compute speed reward based on the velocity of the humanoid
    speed_reward = torch.norm(velocity, p=2, dim=-1)  # Get the speed magnitude
    # Normalize speed reward
    speed_temperature = 1.0  # Temperature for speed normalization
    normalized_speed_reward = torch.exp(speed_temperature * speed_reward)

    # Compute direction reward based on alignment with target direction
    direction_cosine = torch.cos(torch.atan2(target_velocity[:, 1], target_velocity[:, 0]) - 
                                  torch.atan2(velocity[:, 1], velocity[:, 0]))
    direction_reward = (direction_cosine + 1.0) / 2.0  # Normalize to [0, 1]
    # Normalize direction reward
    direction_temperature = 1.0  # Temperature for direction normalization
    normalized_direction_reward = torch.exp(direction_temperature * direction_reward)

    # Total reward is a combination of speed and direction rewards
    total_reward = speed_weight * normalized_speed_reward + direction_weight * normalized_direction_reward

    # Create a dictionary of individual reward components
    reward_components = {
        "speed_reward": normalized_speed_reward,
        "direction_reward": normalized_direction_reward
    }

    return total_reward, reward_components
