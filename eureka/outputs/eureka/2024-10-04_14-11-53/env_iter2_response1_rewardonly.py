@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float, episode_length: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Calculate forward velocity (x direction)
    forward_speed = velocity[:, 0]
    
    # Calculate distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance

    # Parameters
    speed_threshold = 1.0  # Threshold for speed
    distance_reward_scale = 0.5  # Scale for distance reward
    episode_length_factor = 0.1  # Scale for episode length reward
    
    # Define temperature parameters for normalization
    forward_temp: float = 0.1
    distance_temp: float = 0.1
    episode_temp: float = 0.1

    # Reward components
    # Normalize forward speed
    forward_reward = (torch.exp(forward_temp * forward_speed) - 1) * torch.sigmoid(episode_length / 15.0)
    
    # Dynamic distance reward: positive reward for approaching target
    distance_reward = -distance_reward_scale / (distance_to_target + 1e-5)  # Avoid division by zero
    
    # Reward for episode length: encourages longer episodes
    episode_length_reward = episode_length_factor * torch.min(torch.tensor(episode_length), torch.tensor(20.0))

    # Total reward
    total_reward = forward_reward + distance_reward + episode_length_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'distance_reward': distance_reward,
        'episode_length_reward': episode_length_reward
    }

    return total_reward, reward_components
