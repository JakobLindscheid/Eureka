@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature for the velocity reward
    velocity_temp = 0.1

    # Reward for forward velocity (projecting velocity onto the forward direction)
    forward_direction = targets - torso_position
    forward_direction[:, 2] = 0  # Ignore height for direction
    forward_direction_norm = torch.norm(forward_direction, p=2, dim=-1, keepdim=True) + 1e-6  # Avoid divide by zero
    forward_velocity = (velocity * forward_direction).sum(dim=-1) / forward_direction_norm.squeeze()

    # Normalize and scale the forward velocity to a desired range
    reward_velocity = torch.exp(velocity_temp * forward_velocity)
    
    # Penalty for falling or not keeping upright
    upright_penalty = -torch.abs(up_vec[:, 2]) * 10.0  # A larger penalty for falling down

    # Total reward is the sum of the velocity and upright penalties
    total_reward = reward_velocity + upright_penalty

    # Collect individual reward components for further analysis
    individual_rewards = {
        "forward_velocity_reward": reward_velocity,
        "upright_penalty": upright_penalty
    }

    return total_reward, individual_rewards
