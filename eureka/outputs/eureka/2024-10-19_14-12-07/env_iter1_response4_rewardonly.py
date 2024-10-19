@torch.jit.script
def compute_reward(velocity: torch.Tensor, max_velocity: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the reward for the humanoid task, encouraging the humanoid to run as fast as possible.

    Args:
        velocity: A tensor containing the current velocity of the humanoid.
        max_velocity: The maximum achievable velocity for normalization.

    Returns:
        A tuple containing the total reward and a dictionary of individual reward components.
    """

    # Temperature variable for transforming the reward
    temperature = 0.1

    # Compute the speed magnitude
    speed = torch.norm(velocity, p=2, dim=-1)
    
    # Reward is based on how close the speed is to the max velocity
    speed_reward = speed / max_velocity
    speed_reward = torch.exp(temperature * speed_reward) - 1  # Normalize to a fixed range with exp transformation

    # Additional components can be included, for instance:
    # if the humanoid is upright (angle from upright position), we can penalize that
    upright_penalty = -torch.abs(velocity[:, 2])  # Penalize falling or leaning backward

    # Compute the total reward
    total_reward = speed_reward + upright_penalty

    # Create a dictionary for individual reward components for potential logging and analysis
    reward_components = {
        'speed_reward': speed_reward,
        'upright_penalty': upright_penalty,
    }

    return total_reward, reward_components
