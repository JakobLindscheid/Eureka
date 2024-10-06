@torch.jit.script
def compute_reward(root_states: torch.Tensor, velocity: torch.Tensor, target_direction: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the reward based on the ant's forward velocity and direction.

    Args:
        root_states: The states of the ant's body, including position and rotation.
        velocity: The current velocity of the ant.
        target_direction: The desired direction for the ant to move toward.

    Returns:
        total_reward: A scalar reward value.
        reward_components: A dictionary with different components of the reward.
    """
    forward_velocity = velocity[:, 0]  # Assuming forward is along the x-axis
    speed_reward = forward_velocity  # Reward based on speed
    direction_reward = torch.cos(torch.atan2(target_direction[:, 1], target_direction[:, 0]))  # Reward based on direction alignment

    # Normalize rewards using exponential functions
    speed_temp = 0.1  # Temperature for speed reward normalization
    direction_temp = 0.1  # Temperature for direction reward normalization

    speed_reward_normalized = torch.exp(speed_temp * speed_reward)
    direction_reward_normalized = torch.exp(direction_temp * direction_reward)

    total_reward = speed_reward_normalized + direction_reward_normalized

    reward_components = {
        'speed_reward': speed_reward_normalized,
        'direction_reward': direction_reward_normalized,
    }

    return total_reward, reward_components
