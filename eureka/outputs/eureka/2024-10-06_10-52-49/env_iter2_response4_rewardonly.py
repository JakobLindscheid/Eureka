@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.3  # Adjusted to create a sharper reward gradient
    target_distance_temp = 0.2  # Adjusted for improved sensitivity to distance 

    # Extract the torso velocity (x, y, z components)
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Calculate the speed of the torso (magnitude of the velocity vector)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Shape: (N,)
    speed_reward = torch.exp(speed / speed_temp)  # Improved reward scaling for speed

    # Compute distance to the target
    torso_position = root_states[:, 0:3]  # Extract torso position
    to_target = targets - torso_position
    target_distance = torch.norm(to_target, p=2, dim=-1)  # Distance to target
    task_score_reward = torch.exp(-target_distance / target_distance_temp) # Reward for closing distance to target

    # Overall reward combines speed and task score rewards
    total_reward = speed_reward + task_score_reward  # Encourages speed as well as proximity to target

    # Create individual reward components for monitoring
    reward_components = {
        'speed': speed,
        'task_score': -target_distance,  # Still provide insight into distance for monitoring
    }
    
    return total_reward.sum(), reward_components  # Return total reward and individual components
