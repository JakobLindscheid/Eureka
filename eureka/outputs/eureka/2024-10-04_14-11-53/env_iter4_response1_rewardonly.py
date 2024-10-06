@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]
    temperature_forward = 0.2  # Temperature for forward reward scaling
    forward_reward = torch.clamp(torch.exp(forward_speed / temperature_forward) - 1, min=0)  # Apply temperature scaling

    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance

    # Reward for getting closer to the target
    # The closer the distance, the higher the reward
    max_proximity_reward = 10.0  # Set maximum reward obtainable for proximity
    proximity_reward = torch.clamp(max_proximity_reward * (1.0 - distance_to_target / (torch.max(distance_to_target) + 1e-8)), min=0)

    # Total reward
    total_reward = forward_reward + proximity_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward
    }

    return total_reward, reward_components
