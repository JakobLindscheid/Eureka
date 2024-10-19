@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary components from root_states
    velocity = root_states[:, 7:10]  # Get the velocity of the torso
    speed = torch.norm(velocity, p=2, dim=-1)  # Overall speed of the humanoid
    forward_velocity = velocity[:, 0]  # Forward (x) velocity

    # Reward components
    speed_reward = speed ** 2  # Squared reward proportional to overall speed
    direction_bonus = torch.maximum(torch.zeros_like(forward_velocity), forward_velocity) * 2.0  # Forward direction emphasis
    action_penalty = -torch.norm(actions, p=2, dim=-1) ** 3  # Cubed penalty for excessive action usage

    # Temperature parameters for bounding the components
    speed_temp = 0.1  # Increased temperature for speed
    direction_temp = 0.1  # Keep direction-focused
    action_temp = 0.03  # Increased impact on action penalty

    # Transformed rewards
    transformed_speed_reward = torch.exp(speed_temp * speed_reward)  # Normalizing speed reward
    transformed_direction_bonus = torch.exp(direction_temp * direction_bonus)  # Increase variability in direction
    transformed_action_penalty = torch.exp(action_temp * action_penalty)  # More sensitive action penalty

    # Total reward
    total_reward = transformed_speed_reward + transformed_direction_bonus + transformed_action_penalty

    # Create dictionary of individual components
    reward_components = {
        'speed_reward': transformed_speed_reward,
        'direction_bonus': transformed_direction_bonus,
        'action_penalty': transformed_action_penalty
    }

    return total_reward, reward_components
