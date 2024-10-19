@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float, episode_length: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-component)

    # New reward components
    reward_forward_speed = forward_velocity  # Reward based on forward speed
    sustained_speed = torch.max(forward_velocity, torch.tensor(0.0, device=root_states.device))  # Only positive forward velocity matters

    # Small penalty for longer episodes (to encourage efficiency)
    penalty_episode_length = -0.01 * episode_length

    # Temperature for reward transformation
    temperature_forward_speed = 20.0  # Adjusted for better scaling
    temperature_sustained_speed = 10.0

    # Transforming the rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_forward_speed) - 1
    transformed_sustained_speed = torch.exp(sustained_speed / temperature_sustained_speed) - 1

    # Total reward calculation
    total_reward = transformed_reward_forward_speed * 0.5 + transformed_sustained_speed * 1.5 + penalty_episode_length

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'sustained_speed': transformed_sustained_speed,
        'penalty_episode_length': penalty_episode_length,
    }

    return total_reward, reward_components
