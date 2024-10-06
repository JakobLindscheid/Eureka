@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract position and velocity from root states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]          # Get forward velocity along x-axis
    forward_temp = 0.5                       # Temperature for forward reward scaling
    forward_reward = torch.exp(forward_speed / forward_temp) - 1  # Exponential scaling

    # Penalize for not moving (optional)
    not_moving_penalty_temp = 0.1            # Temperature for penalty
    not_moving_penalty = torch.exp(-torch.norm(velocity, p=2, dim=-1) / not_moving_penalty_temp)

    # Total reward combining forward reward and penalty
    total_reward = forward_reward + not_moving_penalty

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'not_moving_penalty': not_moving_penalty
    }

    return total_reward, reward_components
