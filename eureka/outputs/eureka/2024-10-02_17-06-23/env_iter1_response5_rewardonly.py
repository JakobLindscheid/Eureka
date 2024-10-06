@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 1.0  # Adjusted for appropriate scaling
    alignment_temp = 1.0  # Adjusted for appropriate scaling

    # Extract the torso position and velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate the forward velocity component
    forward_velocity = velocity[:, 0]
    
    # Reward component for moving forward quickly, normalized
    speed_reward = forward_velocity / (torch.max(torch.abs(forward_velocity)) + 1e-6)  # Normalize to [0, 1]

    # Calculate the direction to the target
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0  # Ignore the z-axis
    direction_to_target_norm = torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6
    direction_to_target_normalized = direction_to_target / direction_to_target_norm

    # Calculate alignment with the target direction with normalization
    alignment_reward = torch.sum(velocity * direction_to_target_normalized, dim=-1) / (torch.max(torch.norm(velocity, p=2, dim=-1)) + 1e-6)  # Normalize to [0, 1]

    # Total reward is combined
    total_reward = torch.clip(speed_reward + alignment_reward, min=0, max=1)  # Ensure total reward is in [0, 1] range

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
    }
    
    return total_reward, reward_components
