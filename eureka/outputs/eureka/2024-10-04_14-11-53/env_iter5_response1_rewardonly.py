@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (based on x direction velocity)
    forward_speed = velocity[:, 0]
    forward_temp = 1.0  # Temperature parameter for better responsiveness
    forward_reward = torch.exp(forward_speed / forward_temp) - 1  # Exponential scaling for forward movement

    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance to target

    # New proximity reward: Reward for reducing distance to the target
    max_proximity_reward = 10.0  # Maximum reward for being at the target
    proximity_temp = 1.0  # Temperature for proximity scaling
    proximity_reward = torch.exp((distance_to_target - 2.0) / proximity_temp) - 1  # Penalizes moving further away
    proximity_reward = torch.clamp(proximity_reward, min=0.0, max=max_proximity_reward)  # Limit reward within bounds

    # New task score: Encouraging lower distances to target
    task_score = torch.clamp(-distance_to_target / 10.0 + 1.0, min=0.0, max=1.0)  # Normalized positive score

    # Total reward
    total_reward = forward_reward + proximity_reward + task_score

    # Normalizing total reward to fixed range
    total_temp = 1.0  # Overall temperature for reward scaling
    total_reward = torch.exp(total_reward / total_temp) - 1

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
