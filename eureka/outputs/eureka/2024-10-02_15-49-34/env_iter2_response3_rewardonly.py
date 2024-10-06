@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, velocity: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temp parameters for reward scaling
    speed_temp = 0.1
    position_temp = 0.1

    # Extract the torso position and velocity
    torso_position = root_states[:, 0:3]  # Shape: (N, 3)
    torso_velocity = velocity[:, 0:3]      # Shape: (N, 3)

    # Compute distance to target (projected onto the plane)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # Ignore height difference for moving forward
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Shape: (N,)

    # Compute forward speed
    forward_speed = torso_velocity[:, 0]  # Consider forward speed is in the x-direction

    # Reward components
    speed_reward = torch.exp(forward_speed / speed_temp)  # Higher reward for higher speed
    position_penalty = torch.exp(-distance_to_target / position_temp)  # Penalize distance to target

    # Total reward
    total_reward = speed_reward - position_penalty  # Combination of speed reward and positional penalty

    # Create individual component dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'position_penalty': position_penalty,
        'total_reward': total_reward
    }

    return total_reward, reward_components
