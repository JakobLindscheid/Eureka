@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]          # Get forward velocity along x-axis
    forward_temp = 5.0                       # Increased temperature for forward reward scaling
    forward_reward = torch.exp(forward_speed / forward_temp) - 1  # Exponential scaling

    # Distance to target with a positive proximity reward
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance
    max_distance = 10.0                        # Arbitrary maximum distance for normalization
    proximity_temp = 3.0                       # Temperature for proximity reward
    proximity_reward = torch.exp((max_distance - distance_to_target) / proximity_temp) - 1

    # Total reward 
    total_reward = forward_reward + proximity_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward
    }

    return total_reward, reward_components
