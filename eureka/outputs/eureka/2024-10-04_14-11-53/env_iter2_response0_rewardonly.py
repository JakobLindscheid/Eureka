@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Calculate forward speed and normalize
    forward_speed = velocity[:, 0]
    speed_threshold = 3.0  # Define a threshold for good forward speed
    forward_reward = forward_speed / speed_threshold  # Normalize forward speed 

    # Calculate distance to target and reformulate distance reward
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance
    distance_reward = -1.0 / (distance_to_target + 1e-5)  # Reward based on inverse distance to target, add epsilon to avoid division by zero

    # Normalize both components
    total_reward = torch.sigmoid(forward_reward) + torch.sigmoid(distance_reward)

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'distance_reward': distance_reward
    }

    return total_reward, reward_components
