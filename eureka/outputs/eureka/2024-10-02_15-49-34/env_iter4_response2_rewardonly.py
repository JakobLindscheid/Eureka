@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso position and velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Define the target position
    goal_position = targets  

    # Compute distance to the target
    to_target = goal_position - torso_position
    distance = torch.norm(to_target, p=2, dim=-1)

    # Reward for minimizing distance to target (scaled positively)
    distance_reward = 1.0 / (distance + 1.0)  # Inverse distance to give higher rewards when closer

    # Reward for forward speed (x component of velocity)
    forward_speed_reward = velocity[:, 0] * 2.0  # Scale forward speed by a factor of 2

    # Combine rewards
    total_reward = distance_reward + forward_speed_reward

    # Add a penalty for longer episodes
    episode_length_penalty = -0.1 * (1 / dt)  # Encourage shorter episodes
    total_reward += episode_length_penalty

    # Temperature parameters
    temp_distance = 0.5
    temp_speed = 0.5
    
    # Apply temperature scaling
    distance_reward_transformed = torch.exp(distance_reward / temp_distance)
    forward_speed_reward_transformed = torch.exp(forward_speed_reward / temp_speed)

    # Total reward after transformation
    total_reward_transformed = distance_reward_transformed + forward_speed_reward_transformed

    # Create reward components dictionary
    reward_components = {
        'distance_reward': distance_reward,
        'forward_speed_reward': forward_speed_reward,
        'episode_length_penalty': episode_length_penalty
    }
    
    return total_reward_transformed, reward_components
