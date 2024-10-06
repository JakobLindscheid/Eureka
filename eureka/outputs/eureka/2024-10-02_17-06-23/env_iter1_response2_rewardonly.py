@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for scaling rewards
    speed_temp = 0.01
    alignment_temp = 0.01
    forward_penalty = -0.1  # Penalty for not moving towards the target
    
    # Extract the torso position and velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate the forward velocity component (assuming forward is along the x-axis)
    forward_velocity = velocity[:, 0]    

    # Adjust speed reward to be linearly proportional to forward velocity
    speed_reward = forward_velocity * speed_temp  # Now directly linear

    # Calculate direction towards the target with normalized vector
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0  # Ignore the z-axis
    
    # Normalize direction to avoid division by zero
    norm_direction = torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6
    direction_to_target_normalized = direction_to_target / norm_direction

    # Calculate the alignment with respect to the target direction
    alignment_reward = torch.sum(velocity * direction_to_target_normalized, dim=-1) * alignment_temp

    # Introduce a forward movement penalty if the forward velocity is low
    movement_penalty = torch.where(forward_velocity < 0, forward_penalty, 0.0)
    
    # Total reward is a combination of adjusted speed, alignment, and any penalties
    total_reward = speed_reward + alignment_reward + movement_penalty

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'movement_penalty': movement_penalty,
    }
    
    return total_reward, reward_components
