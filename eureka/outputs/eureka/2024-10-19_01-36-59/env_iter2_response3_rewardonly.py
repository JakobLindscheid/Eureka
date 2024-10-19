@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)

    # New reward components
    # Scaling factor to enhance forward speed's influence
    reward_forward_speed = forward_velocity * 2.0  # Scaling up the forward speed reward
    sustained_speed = torch.max(forward_velocity, torch.tensor(0.0, device=root_states.device))

    # Temperature parameters for each component
    temperature_forward = 10.0  # Lower temperature for more sensitivity
    temperature_sustained = 5.0  # More weight to sustained speed

    # Transforming the rewards for scaling
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
