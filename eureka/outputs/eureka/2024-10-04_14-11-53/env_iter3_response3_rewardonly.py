@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]
    forward_reward = torch.clamp(torch.exp(forward_speed) - 1, min=0)

    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance

    # New reward for improving distance to target
    # Reward is based on getting closer to the target in a way that is positively correlated
    # If the distance to target decreases, that should return a positive reward
    successful_proximity_threshold = 2.0  # Define threshold for rewarded proximity
    proximity_reward = torch.where(distance_to_target < successful_proximity_threshold, 
                                   (successful_proximity_threshold - distance_to_target) / successful_proximity_threshold, 
                                   torch.tensor(0.0, device=root_states.device))

    # Total reward
    total_reward = forward_reward + proximity_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward
    }

    return total_reward, reward_components
