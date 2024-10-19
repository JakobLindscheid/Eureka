@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)
    lateral_velocity = torso_velocity[:, 1]   # Lateral velocity (y-axis)
    vertical_velocity = torso_velocity[:, 2]   # Vertical velocity (z-axis)

    # Reward components
    reward_forward_speed = forward_velocity  # Reward is the forward speed
    penalty_lateral_vertical = -torch.norm(lateral_velocity, p=2, dim=-1) - torch.norm(vertical_velocity, p=2, dim=-1)
    
    # Combine both components to form new components
    total_reward = reward_forward_speed + penalty_lateral_vertical

    # Temperature for reward transformation
    temperature_speed = 5.0  # Temperature for forward speed reward
    temperature_penalty = 1.0 # Temperature for penalties

    # Transforming the rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_speed)
    transformed_penalty_lateral_vertical = torch.exp(penalty_lateral_vertical / temperature_penalty)

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_penalty_lateral_vertical

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'penalty_lateral_vertical': transformed_penalty_lateral_vertical,
    }

    return total_reward, reward_components
