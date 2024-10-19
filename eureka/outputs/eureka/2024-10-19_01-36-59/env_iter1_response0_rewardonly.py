@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float, target_speed: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)

    # Reward components
    reward_forward_speed = forward_velocity  # Reward based on forward speed
    reward_consistency = -torch.abs(forward_velocity - target_speed)  # Penalize deviation from target speed

    # Calculate average speed for a more consistent long-term reward
    average_speed = torch.mean(forward_velocity)

    # Define temperatures for reward transformations
    temperature_speed = 5.0
    temperature_consistency = 5.0

    # Transforming rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_speed)
    transformed_reward_consistency = torch.exp(reward_consistency / temperature_consistency)

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_reward_consistency

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'reward_consistency': transformed_reward_consistency,
    }

    return total_reward, reward_components
