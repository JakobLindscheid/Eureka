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

    # Constants for scaling
    speed_threshold = 1.0  # Threshold for meaningful forward speed
    distance_reward_scale = 1.0  # Scale for distance reward
    successful_proximity = 2.0  # Proximity reward when close (less than 2 units)

    # Define temperature parameters for normalization
    speed_temp: float = 0.3
    proximity_temp: float = 0.1

    # Reward components
    # Normalize forward speed
    forward_reward = torch.clamp(torch.exp(forward_speed / speed_threshold) - 1, min=0)

    # Reward for reducing distance to target (more negative distance has less reward)
    proximity_reward = torch.where(distance_to_target < successful_proximity, 
                                   torch.exp((successful_proximity - distance_to_target) / successful_proximity) - 1, 
                                   torch.tensor(0.0, device=root_states.device))

    # Total reward
    total_reward = forward_reward + proximity_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward
    }

    return total_reward, reward_components
