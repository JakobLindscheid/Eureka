@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor,
                   actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis

    # Reward for increasing forward speed over time
    speed_diff = forward_velocity - prev_potentials  # Current speed vs speed from previous frame
    speed_reward = torch.clamp(speed_diff, min=0.0)  # Only reward increases in speed

    # Normalize speed reward to [0, 1] by applying a transformation
    speed_temp = 1.0  # Temperature for the speed reward
    speed_reward = torch.exp(speed_reward / speed_temp)

    # Action Penalty: discourage excessive actions
    action_mag = torch.norm(actions, p=2, dim=-1)  # Calculate the magnitude of actions
    action_penalty = -torch.exp(action_mag / 5.0)  # Temperature for penalty (more severe)

    # Total Reward Calculation
    total_reward = speed_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "speed_reward": speed_reward,
        "action_penalty": action_penalty
    }

    return total_reward, reward_components
