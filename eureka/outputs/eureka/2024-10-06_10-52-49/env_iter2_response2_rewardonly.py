@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.2  # Lower temp to give more reward range for speed
    task_score_temp = 0.5  
    time_penalty_temp = 0.1  # Penalty for longer episodes 

    # Extract the torso velocity (x, y, z components)
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)

    # Calculate the speed of the torso (magnitude of the velocity vector)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Shape: (N,)
    
    # Speed reward with added scaling
    speed_reward = speed * torch.exp(speed / speed_temp)

    # Compute task score as the negative distance to the target but converted to positive scale
    torso_position = root_states[:, 0:3]  # Extract torso position
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Shape: (N,)
    
    # Positive task score based on closeness to the target
    task_score = 1.0 / (1 + distance_to_target)  # Shape: (N,)
    task_score_reward = torch.exp(task_score / task_score_temp)

    # Time penalty for episodes that did not reach the target quickly enough
    episode_length = dt # This represents time spent (replace this with actual episode length if available)
    time_penalty = -episode_length * time_penalty_temp  # Penalty for time spent not reaching goal

    # Overall reward combines speed and task score rewards minus time penalty
    total_reward = speed_reward + task_score_reward + time_penalty

    # Create individual reward components for monitoring
    reward_components = {
        'speed': speed,
        'task_score': distance_to_target,
        'time_penalty': time_penalty,
    }

    return total_reward.sum(), reward_components  # Return total reward and individual components
