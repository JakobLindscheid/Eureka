@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float, episode_length: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Calculate the forward movement (x direction) and its speed
    forward_speed = velocity[:, 0]
    
    # Define temperature parameters for normalization
    forward_temp: float = 0.2  # Increased temperature for forward movement effectiveness
    efficiency_temp: float = 0.05  # Temperature for efficiency reward

    # Reward components
    forward_reward = torch.exp(forward_temp * forward_speed)  # Reward for moving forward
    
    # New efficiency reward: negative reward for longer episodes, encouraging completion
    efficiency_reward = -torch.exp(efficiency_temp * episode_length)

    # Total reward
    total_reward = forward_reward + efficiency_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'efficiency_reward': efficiency_reward
    }

    return total_reward, reward_components
