@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Calculate the forward movement (x direction)
    forward_speed = velocity[:, 0]
    
    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance

    # Constants (you may need to adjust these further based on experimentation)
    speed_threshold = 1.0  # Define a threshold for good forward speed
    distance_reward_scale = 0.1  # Scale for distance reward

    # Define temperature parameters for normalization
    speed_temp: float = 0.5
    distance_temp: float = 0.1
    
    # Reward components
    # Normalize forward speed and provide reward based on a threshold
    forward_reward = torch.clamp(torch.exp(forward_speed / speed_threshold) - 1, min=0)  # Clamping to avoid negative rewards
    
    # Provide a reward based on inverse distance to target
    distance_reward = -distance_reward_scale * distance_to_target
    
    # Total reward
    total_reward = forward_reward + torch.exp(distance_temp * distance_reward)

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'distance_reward': distance_reward
    }

    return total_reward, reward_components
