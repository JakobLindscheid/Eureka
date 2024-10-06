@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the reward for the ant running forward on a tilted plane.

    Args:
        torso_position: The current position of the torso of the ant.
        velocity: The current velocity of the ant.
        target_position: The target position the ant should ideally move towards.

    Returns:
        Tuple containing total reward and a dictionary of individual reward components.
    """
    # Constants for reward components
    forward_reward_temp = 0.1  # Temperature for forward velocity reward
    distance_reward_temp = 0.2  # Temperature for distance reward
    
    # Calculate forward speed reward (reward high for higher velocity in the forward direction)
    forward_velocity = velocity[:, 0]  # Assuming forward direction is the x-axis
    forward_reward = torch.mean(forward_velocity)

    # Calculate distance to target (reward high for being closer to the target)
    distance_to_target = torch.norm(target_position - torso_position, p=2, dim=-1)
    distance_reward = -torch.mean(distance_to_target)  # Negative reward for farther away

    # Normalize components by applying exponential transformation
    forward_reward_normalized = torch.exp(forward_reward_temp * forward_reward)
    distance_reward_normalized = torch.exp(distance_reward_temp * distance_reward)

    # Total reward is a combination of both components
    total_reward = forward_reward_normalized + distance_reward_normalized

    # Create a dictionary for individual reward components
    reward_components = {
        'forward_reward': forward_reward_normalized,
        'distance_reward': distance_reward_normalized
    }

    return total_reward, reward_components
