@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)
    
    # New reward components
    reward_forward_speed = forward_velocity ** 2  # Squared forward velocity
    sustained_speed = torch.max(forward_velocity, torch.tensor(0.0, device=root_states.device)) ** 2  # Only positive forward velocity matters

    # Temperature for reward transformation
    temperature_speed = 200.0  # Increased for wider range and stability
    temperature_sustained_speed = 50.0  # Increased for improved scaling

    # Transforming the rewards for better scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_speed) - 1  # Shifted to be more normalized
    transformed_sustained_speed = torch.exp(sustained_speed / temperature_sustained_speed) - 1  # Shifted as well

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_sustained_speed

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'sustained_speed': transformed_sustained_speed,
    }

    return total_reward, reward_components
