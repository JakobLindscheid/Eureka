@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants for reward shaping
    forward_reward_scale: float = 1.0  # Scale for the forward movement reward
    velocity_reward_scale: float = 0.1  # Scale for the speed reward
    temperature: float = 0.5  # Temperature variable for reward transformation

    # Calculate the forward direction (based on the second axis of torso_position)
    forward_direction = torch.tensor([1.0, 0.0, 0.0], device=torso_position.device)

    # Calculate the positional distance to the target
    distance_to_target = target_position - torso_position
    forward_distance = torch.dot(distance_to_target, forward_direction)

    # Calculate rewards
    forward_reward = forward_reward_scale * forward_distance  # Reward for moving towards the target
    speed_reward = velocity.norm(p=2, dim=-1)  # Reward based on speed
    total_reward = forward_reward + velocity_reward_scale * speed_reward

    # Normalize rewards using an exponential function
    transformed_forward_reward = torch.exp(forward_reward / temperature)
    transformed_speed_reward = torch.exp(speed_reward / temperature)

    # Compile individual reward components
    reward_components = {
        'forward_reward': transformed_forward_reward,
        'speed_reward': transformed_speed_reward,
    }

    return total_reward, reward_components
