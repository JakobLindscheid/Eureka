@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 0.3
    alignment_temp = 0.3
    target_reach_temp = 0.5
    episode_length_bonus_temp = 0.1

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component
    forward_velocity = velocity[:, 0]

    # Reward component for moving forward quickly (normalized)
    speed_reward = torch.exp(forward_velocity / speed_temp)

    # Calculate the direction towards the target
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0  # Ignore z-axis
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1) + 1e-6  # Avoid division by zero

    # Reward for reducing distance to the target
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)

    # Alignment with target direction
    direction_to_target_normalized = direction_to_target / distance_to_target.unsqueeze(-1)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / 10.0, min=0.0)  # Normalize alignment

    # Bonus for achieving longer episodes
    episode_length_bonus = torch.exp(dt / episode_length_bonus_temp)  # Use time step duration as a proxy for episode length

    # Combine all rewards
    total_reward = speed_reward + alignment_reward + target_reach_reward + episode_length_bonus

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'target_reach_reward': target_reach_reward,
        'episode_length_bonus': episode_length_bonus,
    }
    
    return total_reward, reward_components
