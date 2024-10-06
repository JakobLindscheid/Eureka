@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 1.0
    alignment_temp = 1.0

    # Extract torso position and linear velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate forward velocity component
    forward_velocity = velocity[:, 0]  # Assuming forward is in x-direction
    forward_velocity_norm = torch.norm(velocity, p=2, dim=1)

    # Reward component for moving forward (normalized)
    speed_reward = torch.exp(forward_velocity / (speed_temp + 1e-6)) * (forward_velocity_norm + 1e-6)

    # Calculate directional alignment towards target
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0  # Ignore vertical direction
    direction_to_target_magnitude = torch.norm(direction_to_target, p=2, dim=1, keepdim=True) + 1e-6
    direction_to_target_normalized = direction_to_target / direction_to_target_magnitude

    # Alignment reward based on how well the walking direction aligns with the target direction
    alignment_reward = torch.exp(torch.sum(velocity * direction_to_target_normalized, dim=1) / (alignment_temp + 1e-6))

    # Combine rewards for total reward
    total_reward = speed_reward + alignment_reward

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
    }
    
    return total_reward, reward_components
