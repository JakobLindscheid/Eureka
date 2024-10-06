@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.2
    task_score_temp = 0.1

    # Extract torso velocity (x, y, z components)
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Calculate the speed of the torso (magnitude of the velocity vector)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Shape: (N,)
    
    # Reward based on speed with a quadratic form
    speed_reward = speed ** 2 / (speed_temp + 1e-6)  # Avoid division by zero

    # Compute task score as the negative distance to the target mapped positively
    torso_position = root_states[:, 0:3]  # Extract torso position
    to_target = targets - torso_position
    dist_to_target = torch.norm(to_target, p=2, dim=-1)  # Distance to target
    task_score_reward = -dist_to_target  # Negative distance as reward

    # Add a small offset to shift mean values above zero to improve learning
    task_score_reward = task_score_reward + 1000.0

    # Linear reward for the distance covered in the task
    distance_covered_reward = torch.exp(-(dist_to_target / 10.0))

    # Overall reward combines speed, task score, and linear distance covered rewards
    total_reward = speed_reward + task_score_reward + distance_covered_reward

    # Create individual reward components for monitoring
    reward_components = {
        'speed': speed,
        'task_score': task_score_reward,
        'distance_covered': distance_covered_reward,
    }
    
    return total_reward.sum(), reward_components  # Return total reward and individual components
