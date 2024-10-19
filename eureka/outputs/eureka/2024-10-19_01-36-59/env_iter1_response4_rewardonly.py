@torch.jit.script
def compute_reward(root_states: torch.Tensor, episode_length: float, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the torso's velocity from the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)

    # Reward components
    reward_forward_speed = forward_velocity  # Reward directly based on the forward speed
    
    # Stability Reward: encouraging the agent to maintain its forward speed over time
    stability_reward = torch.clamp(forward_velocity / (0.1 + dt), min=0.0)  # Normalize by time to encourage consistent speeds

    # Episode length reward: encourage running longer
    reward_episode_length = episode_length / 1000.0  # Scale the episode length reward

    # Temperature for reward transformation
    temperature_speed = 2.0
    temperature_stability = 1.0
    temperature_episode_length = 1.0

    # Transforming the rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_speed)
    transformed_stability_reward = torch.exp(stability_reward / temperature_stability)
    transformed_reward_episode_length = torch.exp(reward_episode_length / temperature_episode_length)

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_stability_reward + transformed_reward_episode_length

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'reward_stability': transformed_stability_reward,
        'reward_episode_length': transformed_reward_episode_length,
    }

    return total_reward, reward_components
