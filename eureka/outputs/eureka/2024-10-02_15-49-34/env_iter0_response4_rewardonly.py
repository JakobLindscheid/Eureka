@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, targets: torch.Tensor, 
                   dt: float, tilt_angle: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    forward_reward_weight = 1.0
    tilt_penalty_weight = 0.5

    # Compute the forward component of the velocity
    forward_velocity = velocity[:, 0]  # Assuming the first column is the forward direction (x-axis)

    # Compute the angle between the torso orientation and the tilt direction
    tilt_direction = torch.tensor([torch.cos(tilt_angle), 0, torch.sin(tilt_angle)], device=torso_position.device)
    angle_to_tilt = torch.atan2(torso_position[:, 2] - tilt_direction[2], torso_position[:, 0] - tilt_direction[0])

    # Reward for forward velocity
    forward_reward = forward_reward_weight * torch.exp(forward_velocity)

    # Penalty for tilting too much
    tilt_penalty = tilt_penalty_weight * torch.exp(-torch.abs(angle_to_tilt))

    # Total reward
    total_reward = forward_reward - tilt_penalty

    # Create a dictionary for the reward components
    reward_components = {
        "forward_reward": forward_reward,
        "tilt_penalty": tilt_penalty,
    }

    return total_reward, reward_components
