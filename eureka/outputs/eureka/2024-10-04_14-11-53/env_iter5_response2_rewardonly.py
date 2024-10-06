@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]
    forward_temp = 0.5  # Reduced temperature parameter for forward reward scaling
    forward_reward = torch.exp(forward_speed / forward_temp) - 1  # Exponential scaling

    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance
    
    # Revised proximity reward: continuous reward based on distance from target
    max_proximity_reward = 10.0  # Max reward for minimizing distance
    proximity_reward = torch.clamp((2.0 - distance_to_target) / 2.0 * max_proximity_reward, min=0.0)

    # Revised task score: avoiding negatives
    task_score = torch.clamp(1.0 - (distance_to_target / 10.0), min=0.0, max=1.0)

    # Total reward
    total_reward = forward_reward + proximity_reward + task_score

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
