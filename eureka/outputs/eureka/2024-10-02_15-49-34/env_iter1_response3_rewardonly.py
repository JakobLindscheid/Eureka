@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    forward_direction = torch.tensor([1.0, 0.0, 0.0], device=torso_position.device)
    temperature_position = 0.1
    temperature_velocity = 0.1

    # Compute the distance to the target
    distance_to_target = torch.norm(torso_position - target_position, p=2, dim=-1)

    # Reward component for getting closer to the target
    position_reward = -distance_to_target
    position_reward_normalized = torch.exp(temperature_position * position_reward)

    # Reward component for forward velocity
    forward_velocity = torch.sum(velocity * forward_direction, dim=-1)
    velocity_reward = forward_velocity
    velocity_reward_normalized = torch.exp(temperature_velocity * velocity_reward)

    # Total reward
    total_reward = position_reward_normalized + velocity_reward_normalized

    # Create an individual rewards dictionary
    reward_components = {
        'position_reward': position_reward_normalized,
        'velocity_reward': velocity_reward_normalized
    }

    return total_reward, reward_components
