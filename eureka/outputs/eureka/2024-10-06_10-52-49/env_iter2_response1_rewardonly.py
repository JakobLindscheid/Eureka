@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.3
    task_score_temp = 0.5
    distance_penalty_temp = 1.0

    # Extract torso position and velocity
    torso_position = root_states[:, 0:3]
    torso_velocity = root_states[:, 7:10]
    
    # Calculate speed of the torso (magnitude of the velocity vector)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Shape: (N,)
    
    # Define speed threshold
    speed_threshold = 0.5
    speed_reward = torch.where(speed > speed_threshold, torch.exp(speed / speed_temp), -torch.abs(speed - speed_threshold))

    # Compute distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Shape: (N,)
    
    # Transform distance to target into a positive task score
    task_score = -distance_to_target  # Closer is better, reward is positive
    task_score_reward = torch.exp(task_score / task_score_temp)  # Normalize with temperature

    # Penalize episode length if distance to target is not decreasing
    distance_penalty = -torch.log1p(distance_to_target / 10.0)  # Normalize penalty
    distance_penalty = distance_penalty / distance_penalty_temp

    # Overall reward combines speed and task score rewards, adjusting for the length of the episode
    total_reward = speed_reward + task_score_reward + distance_penalty

    # Create individual reward components for monitoring
    reward_components = {
        'speed': speed,
        'task_score': task_score,
        'distance_penalty': distance_penalty,
    }
    
    return total_reward.sum(), reward_components  # Return total reward and individual components
