@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward speed reward (x direction)
    forward_speed = velocity[:, 0]
    
    # Hyperparameters for scaling rewards
    forward_temp = 0.5  # Temperature for forward speed reward transformation
    proximity_temp = 0.5  # Temperature for proximity reward transformation

    # Reward based on forward speed
    forward_reward = torch.exp(forward_temp * forward_speed) - 1
    
    # Calculate distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance
    desired_proximity = 2.0  # Threshold for proximity reward

    # Proximity reward (encouraging getting closer to the target)
    proximity_reward = torch.clamp((desired_proximity - distance_to_target) / desired_proximity, min=0)

    # New task score based on a combined function of forward movement and proximity improvement
    task_score = (forward_speed - distance_to_target).clamp(min=0)  # Reward when moving closely towards the target.

    # Total reward
    total_reward = forward_reward + proximity_reward + task_score

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
