@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 0.1
    alignment_temp = 0.1
    target_reach_temp = 0.5
    inactivity_penalty_temp = 0.3

    # Extract necessary components
    torso_position = root_states[:, 0:3]  # Torso position
    velocity = root_states[:, 7:10]  # Velocity of the torso
    
    # Forward velocity component
    forward_velocity = velocity[:, 0]
    
    # Reward component for moving forward quickly (boosted scaling)
    speed_reward = torch.clamp(forward_velocity * 2.0, min=0.0)  # Increased scaling
    
    # Calculate the direction toward the target
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0  # Ignore z-axis for 2D movement
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)

    # Improved reward for covering ground towards the target
    distance_reward = torch.exp(-distance_to_target / target_reach_temp)  # Exponential decay based on distance

    # Reward for aligning the torso direction with the target direction
    direction_to_target_normalized = torch.nn.functional.normalize(direction_to_target, dim=-1)
    forward_direction_normalized = torch.nn.functional.normalize(velocity, dim=-1)
    alignment_reward = torch.sum(forward_direction_normalized * direction_to_target_normalized, dim=-1)  # Dot product

    # Inactivity penalty if the forward velocity is very low
    inactivity_penalty = torch.exp(-torch.clamp(forward_velocity, min=0.0, max=1.0) / inactivity_penalty_temp)

    # Total reward is the combination of all components
    total_reward = speed_reward + alignment_reward + distance_reward - inactivity_penalty

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'distance_reward': distance_reward,
        'inactivity_penalty': inactivity_penalty,
    }
    
    return total_reward, reward_components
