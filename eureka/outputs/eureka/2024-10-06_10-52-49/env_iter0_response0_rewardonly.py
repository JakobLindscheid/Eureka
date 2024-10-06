@torch.jit.script
def compute_reward(velocity: torch.Tensor, target_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature for reward component transformations
    velocity_temp = 0.1

    # Compute the speed of the humanoid
    speed = torch.norm(velocity, p=2, dim=-1)

    # Compute the reward based on speed; we want the humanoid to maximize speed
    speed_reward = speed.unsqueeze(-1)

    # Normalize the reward using an exponential function
    normalized_speed_reward = torch.exp(speed_reward / velocity_temp)

    # The goal is to run as fast as possible, can also encourage alignment with target velocity
    direction_reward = torch.dot(velocity, target_velocity) / (torch.norm(velocity) + 1e-6)  # avoid division by zero
    direction_reward = direction_reward.unsqueeze(-1)  # make sure it's in the shape of (n, 1)

    # Combine the rewards for a total reward
    total_reward = normalized_speed_reward + direction_reward

    # Build the dictionary of individual reward components
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'direction_reward': direction_reward
    }

    return total_reward, reward_components
