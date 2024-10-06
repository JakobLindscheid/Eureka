@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Calculate the forward movement (x direction)
    forward_speed = velocity[:, 0]  # Assuming forward movement is along the x-axis
    
    # Calculate distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)

    # Define temperature parameters for normalization
    forward_temp: float = 0.05
    distance_temp: float = 0.1
    speed_temp: float = 0.1

    # Reward components
    forward_reward = forward_speed         # Removing the exp as it saturates
    distance_reward = -distance_to_target   # Penalizing distance
    speed_reward = torch.exp(speed_temp * (torch.norm(velocity, p=2, dim=-1)))  # Reward for maintaining speed

    # Total reward
    total_reward = forward_reward + distance_reward + speed_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'distance_reward': distance_reward,
        'speed_reward': speed_reward,
    }

    return total_reward, reward_components
