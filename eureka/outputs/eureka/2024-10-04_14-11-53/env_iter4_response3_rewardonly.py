@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]
    forward_temp = 1.0  # temperature parameter for forward reward scaling
    forward_reward = torch.exp(forward_speed / forward_temp) - 1  # Exponential scaling

    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance

    # New proximity reward: provide reward based on distance from target
    max_proximity_reward = 10.0  # Maximum reward for being at target
    successful_proximity_threshold = 2.0
    proximity_reward = torch.where(distance_to_target < successful_proximity_threshold, 
                                   (successful_proximity_threshold - distance_to_target) / successful_proximity_threshold * max_proximity_reward, 
                                   torch.tensor(0.0, device=root_states.device))

    # Total reward
    total_reward = forward_reward + proximity_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward
    }

    return total_reward, reward_components
