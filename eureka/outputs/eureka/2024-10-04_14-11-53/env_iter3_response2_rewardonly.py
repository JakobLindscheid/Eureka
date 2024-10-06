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

    # Define constants for scaling and thresholds
    speed_threshold = 1.0  # Threshold for meaningful forward speed
    successful_proximity = 2.0  # Reward for being close to the target

    # Define temperature parameters for normalization
    speed_temp: float = 0.5
    distance_temp: float = 0.3

    # Reward components
    # Normalize forward speed
    forward_reward = torch.clamp(torch.exp(forward_speed / speed_temp) - 1, min=0)

    # Reward for reducing distance to target
    proximity_reward = torch.where(distance_to_target < successful_proximity,
                                   (successful_proximity - distance_to_target) / successful_proximity,
                                   torch.tensor(0.0, device=root_states.device))

    # Calculate total reward using both components
    total_reward = forward_reward + proximity_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward
    }

    return total_reward, reward_components
