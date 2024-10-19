@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)
    
    # New reward components
    reward_forward_speed = forward_velocity  # Reward based on forward speed
    sustained_speed = torch.max(forward_velocity, torch.tensor(0.0, device=root_states.device))  # Only positive forward velocity matters

    # Temperature for reward transformation
    temperature_speed = 50.0  # Change this for stability
    temperature_sustained_speed = 10.0

    # Transforming the rewards for scaling
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
