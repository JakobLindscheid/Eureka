@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)

    # Reward components
    reward_forward_speed = forward_velocity  # Reward is the forward speed
    reward_penalty = -torch.norm(torso_velocity, p=2, dim=-1)  # Penalize based on total velocity

    # Temperature for reward transformation
    temperature_speed = 1.0
    temperature_penalty = 1.0

    # Transforming the rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_speed)
    transformed_reward_penalty = torch.exp(reward_penalty / temperature_penalty)

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_reward_penalty

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'reward_penalty': transformed_reward_penalty,
    }

    return total_reward, reward_components
