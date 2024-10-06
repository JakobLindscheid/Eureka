@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters
    speed_temp = 1.0
    alignment_temp = 1.0
    stationary_penalty = 0.01

    # Extract torso position and velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate forward velocity
    forward_velocity = velocity[:, 0]
    
    # Normalized speed reward
    normalized_speed = torch.clamp(forward_velocity, 0, 10)  # Limit speed to max 10 units
    speed_reward = normalized_speed / 10.0  # Scale to [0, 1]

    # Calculate direction to target
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0
    direction_to_target_norm = torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6
    direction_to_target_normalized = direction_to_target / direction_to_target_norm

    # Compute alignment based on cosine of the angle
    alignment_score = torch.sum(velocity * direction_to_target_normalized, dim=-1) / (torch.norm(velocity, p=2, dim=-1) + 1e-6)
    alignment_reward = torch.clamp(alignment_score, 0, 1)  # Scale to [0, 1]

    # Penalty for not moving forward
    if forward_velocity < stationary_penalty:
        stationary_penalty_reward = -1.0
    else:
        stationary_penalty_reward = 0.0

    # Total reward
    total_reward = speed_reward + alignment_reward + stationary_penalty_reward

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'stationary_penalty': stationary_penalty_reward,
    }
    
    return total_reward, reward_components
