@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)
    lateral_velocity = torso_velocity[:, 1:2]  # Lateral velocity (y-axis)

    # Reward components
    reward_forward_speed = forward_velocity  # Reward is the forward speed
    reward_stability = -torch.abs(lateral_velocity)  # Reward stability based on lateral movement

    # Temperature for reward transformation
    temperature_forward_speed = 10.0
    temperature_stability = 1.0

    # Transforming the rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_forward_speed)
    transformed_reward_stability = torch.exp(reward_stability / temperature_stability)

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_reward_stability

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'reward_stability': transformed_reward_stability,
    }

    return total_reward, reward_components
