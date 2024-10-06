@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 2.0
    alignment_temp = 2.0
    distance_temp = 1.0

    # Extract position, velocity
    torso_position = root_states[:, 0:3]
    torso_velocity = root_states[:, 7:10]

    # Reward component for moving forward quickly (normalize within a range)
    forward_velocity = torso_velocity[:, 0]
    speed_reward = torch.clamp(forward_velocity / 5.0, min=0.0, max=1.0)

    # Calculate the direction to target
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0  # Project to 2D plane
    norm_direction = torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6
    direction_to_target_normalized = direction_to_target / norm_direction

    # Calculate the alignment of the torso velocity with the target direction
    alignment_reward = torch.mean(torch.sigmoid(torch.sum(torso_velocity * direction_to_target_normalized, dim=-1)))

    # Distance penalty for not being close to the target (this encourages moving towards it)
    distance_penalty = torch.clamp(norm_direction / 10.0, min=0.0, max=1.0)

    # Combine rewards into total reward
    total_reward = speed_reward + alignment_reward - distance_penalty

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'distance_penalty': distance_penalty,
    }
    
    return total_reward, reward_components
