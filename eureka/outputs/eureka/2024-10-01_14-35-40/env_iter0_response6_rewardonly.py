@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature for reward transformation
    temp_forward_speed = 0.1

    # Reward component for forward speed
    forward_speed = velocity[:, up_axis_idx].clone()  # Extract forward speed based on up_axis
    reward_forward_speed = torch.exp(forward_speed * temp_forward_speed)

    # Total reward
    total_reward = reward_forward_speed.mean()  # Average over the batch

    # Create a dictionary for individual reward components
    reward_components = {
        'forward_speed': reward_forward_speed,
    }

    return total_reward, reward_components
