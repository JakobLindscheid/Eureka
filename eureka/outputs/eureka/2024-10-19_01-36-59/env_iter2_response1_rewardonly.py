@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)
    
    # Ensure we are using positive values only
    forward_velocity = torch.clamp(forward_velocity, min=0.0)

    # Reward components
    reward_forward_speed = forward_velocity ** 2  # Squared to emphasize higher speeds
    sustained_speed = torch.mean(forward_velocity)  # Average forward speed over the episode

    # Temperature for reward transformation
    temperature_forward = 5.0  # Tuning for forward speed
    temperature_sustained = 20.0  # Tuning for sustained speed

    # Transforming rewards for more balanced feedback
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_forward) - 1
    transformed_sustained_speed = torch.exp(sustained_speed / temperature_sustained) - 1

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_sustained_speed

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'sustained_speed': transformed_sustained_speed,
    }

    return total_reward, reward_components
