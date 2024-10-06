@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming the reward components.
    speed_temp = 0.1
    alignment_temp = 0.1
    target_reach_temp = 0.1
    episode_length_temp = 0.05

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Forward velocity (scaled for larger gradients)
    forward_velocity = velocity[:, 0]
    speed_reward = torch.exp(forward_velocity / speed_temp)
    
    # Calculate the direction toward the target
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0  # Ignore z-axis
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)

    # Proximity to target
    target_reach_reward = -torch.exp(-distance_to_target / target_reach_temp)  # Reward for being closer to the target

    # Normalizing alignment reward
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1), min=0.0)
    alignment_reward = torch.exp(alignment_reward / alignment_temp)  # Transform for better scaling

    # Episode length reward proportionate to the time taken (potentially lengthen for continuous movement)
    episode_length_reward = torch.exp(torch.mean(forward_velocity) / episode_length_temp)

    # Combine all reward components
    total_reward = speed_reward + alignment_reward + target_reach_reward + episode_length_reward

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'target_reach_reward': target_reach_reward,
        'episode_length_reward': episode_length_reward,
    }

    return total_reward, reward_components
