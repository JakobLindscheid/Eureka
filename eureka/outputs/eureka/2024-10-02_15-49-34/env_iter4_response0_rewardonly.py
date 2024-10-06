@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso position and velocity
    torso_position = root_states[:, 0:3]  # Shape: (batch_size, 3)
    velocity = root_states[:, 7:10]  # Shape: (batch_size, 3)

    # Define the target position (goal in x-direction, keep y, z same)
    goal_position = targets  # Targets are predefined in the environment

    # Compute distance to the target
    to_target = goal_position - torso_position
    distance = torch.norm(to_target, p=2, dim=-1)  # Distance to target

    # Reward for minimizing distance to target (encouraging moving forward)
    distance_reward = -distance / 10.0  # Normalized to have a meaningful scale

    # Reward for forward speed (x component of velocity)
    forward_speed_reward = velocity[:, 0] * 2.0  # Scale up the forward speed component

    # Task score - reward for maintaining velocity (magnitude of speed)
    task_score_reward = torch.norm(velocity, p=2, dim=-1) / 5.0  # Encourage overall speed

    # Combine rewards
    total_reward = distance_reward + forward_speed_reward + task_score_reward
    
    # Temperature parameters
    temp_distance = 1.0
    temp_speed = 1.0
    temp_task_score = 1.0
    
    # Apply temperature scaling
    distance_reward_transformed = torch.exp(distance_reward / temp_distance)
    forward_speed_reward_transformed = torch.exp(forward_speed_reward / temp_speed)
    task_score_reward_transformed = torch.exp(task_score_reward / temp_task_score)

    # Total reward after transformation
    total_reward_transformed = distance_reward_transformed + forward_speed_reward_transformed + task_score_reward_transformed
    
    # Create reward components dictionary
    reward_components = {
        'distance_reward': distance_reward,
        'forward_speed_reward': forward_speed_reward,
        'task_score_reward': task_score_reward
    }
    
    return total_reward_transformed, reward_components
