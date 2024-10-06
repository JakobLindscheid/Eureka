@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]          # Get forward velocity along x-axis
    forward_temp = 2.0                       # Adjusted temperature for forward reward scaling
    forward_reward = torch.exp(forward_speed / forward_temp) - 1  # Exponential scaling

    # Distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance
    max_distance = 10.0                        # Maximum distance for normalization
    task_score_temp = 5.0                      # Temperature for the task score
    task_score = torch.exp((max_distance - distance_to_target) / task_score_temp) - 1 

    # Total reward
    total_reward = forward_reward + task_score

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
