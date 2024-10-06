@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]
    forward_reward_temp = 1.0  # Temperature for forward reward scaling
    forward_reward = torch.exp(forward_speed / forward_reward_temp) - 1.0

    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance

    # Stability and speed towards goal
    stability_reward = torch.clip(torch.dot(velocity, to_target) / (torch.norm(velocity) + 1e-5), 0, 1)

    # New task score that incorporates both forward movement and stability
    task_score_temp = 0.5  # Temperature for task score component
    task_score = torch.exp(stability_reward / task_score_temp) - 1.0

    # Total reward
    total_reward = forward_reward + task_score

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
