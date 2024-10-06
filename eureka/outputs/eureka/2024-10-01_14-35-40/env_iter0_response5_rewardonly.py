@torch.jit.script
def compute_reward(torso_position: torch.Tensor, torso_velocity: torch.Tensor, target_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    forward_direction = torch.tensor([1.0, 0.0, 0.0], device=torso_position.device)  # Forward along x-axis

    # Reward components
    velocity_magnitude = torch.norm(torso_velocity, p=2, dim=-1)  # Magnitude of the torso's velocity
    direction_component = torch.matmul(torso_velocity, forward_direction)  # Dot product to get forward velocity

    # Temperature parameters
    temperature_velocity = 0.01
    temperature_direction = 0.01
    
    # Compute normalized rewards
    forward_reward = torch.exp(direction_component / temperature_direction)
    speed_reward = torch.exp(velocity_magnitude / temperature_velocity)

    # Total reward
    total_reward = forward_reward + speed_reward

    # Reward components dictionary
    reward_components = {
        'forward_reward': forward_reward,
        'speed_reward': speed_reward,
    }

    return total_reward, reward_components
