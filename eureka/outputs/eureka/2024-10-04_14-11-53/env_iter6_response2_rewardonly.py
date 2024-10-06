@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (velocity in x direction)
    forward_speed = velocity[:, 0]
    forward_temp = 0.5  # Reduced temperature for more sensitivity
    forward_reward = torch.exp(forward_speed / forward_temp) - 1  # Exponential scaling

    # Progress Reward: positive feedback based on minimizing distance to the target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance
    max_distance = 10.0  # Adjust for performance scaling
    progress_temp = 1.0  # Temperature for progress reward scaling
    # Reward based on the inverse of distance for close to the target position
    progress_reward = torch.exp((max_distance - distance_to_target) / progress_temp) - 1

    # Task Score: positive reward when making significant reduction in target distance
    safe_distance_threshold = 2.0  # When close enough to target
    task_score = torch.where(distance_to_target < safe_distance_threshold, 
                             max_distance - distance_to_target, 
                             torch.tensor(0.0, device=root_states.device))

    # Total reward
    total_reward = forward_reward + progress_reward + task_score

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'progress_reward': progress_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
