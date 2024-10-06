@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.5
    task_score_temp = 0.1

    # Extract the torso velocity (x, y, z components)
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Calculate the speed of the torso (magnitude of the velocity vector)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Shape: (N,)

    # Reward based on speed with normalization
    speed_reward = torch.exp(speed / speed_temp)

    # Compute task score as the negative distance to the target
    torso_position = root_states[:, 0:3]  # Extract torso position
    to_target = targets - torso_position
    task_score = -torch.norm(to_target, p=2, dim=-1)  # Shape: (N,)
    
    # Normalizing the task score
    task_score_reward = torch.exp(task_score / task_score_temp)

    # Overall reward combines speed and task score rewards
    total_reward = speed_reward + task_score_reward  # Encourages speed as well as proximity to target

    # Create individual reward components for monitoring
    reward_components = {
        'speed': speed,
        'task_score': task_score,
    }
    
    return total_reward.sum(), reward_components  # Return total reward and individual components
