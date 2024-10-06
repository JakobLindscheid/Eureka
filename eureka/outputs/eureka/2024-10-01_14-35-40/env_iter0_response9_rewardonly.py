@torch.jit.script
def compute_reward(torso_position: torch.Tensor, torso_velocity: torch.Tensor, target_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    REWARD_SCALE = 1.0
    VELOCITY_TEMP = 0.5  # Temperature for velocity reward normalization
    
    # Forward direction (assuming positive x direction)
    forward_direction = torch.tensor([1.0, 0.0, 0.0], device=torso_position.device)

    # Compute the dot product of the torso velocity with the forward direction
    forward_velocity = torch.dot(torso_velocity, forward_direction)

    # Reward is proportional to the forward velocity
    velocity_reward = REWARD_SCALE * forward_velocity
    # Normalize using exponential transformation
    transformed_velocity_reward = torch.exp(VELOCITY_TEMP * velocity_reward)

    # Main reward is the transformed velocity reward
    total_reward = transformed_velocity_reward

    # Create reward components dictionary
    reward_components = {
        'velocity_reward': transformed_velocity_reward,
    }

    return total_reward, reward_components
