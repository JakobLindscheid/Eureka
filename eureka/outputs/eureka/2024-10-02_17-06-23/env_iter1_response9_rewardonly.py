@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 1.0
    alignment_temp = 0.1
    misalignment_penalty_temp = 10.0

    # Extract the torso position and velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate the forward velocity component (assuming forward is along the x-axis)
    forward_velocity = velocity[:, 0]
    
    # Reward component for moving forward quickly (squared velocity to better penalize slow movement)
    speed_reward = torch.exp((forward_velocity ** 2) / speed_temp)

    # Calculate the direction towards the target
    direction_to_target = targets - torso_position
    direction_to_target[direction_to_target[:, 2] != 0, 2] = 0.0  # Ignore the z-axis for forward movement
    direction_to_target_normalized = direction_to_target / (torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6)

    # Alignment reward based on cosine similarity component and an additional penalty for misalignment
    alignment_reward = torch.sum(velocity * direction_to_target_normalized, dim=-1)
    misalignment_penalty = -torch.abs(torch.sum(alignment_reward) / misalignment_penalty_temp)
    
    # Normalize alignment reward using an exponential transformation
    alignment_reward_transformed = torch.exp(alignment_reward / alignment_temp)

    # Total reward combines speed reward, normalized alignment reward and penalties
    total_reward = speed_reward + alignment_reward_transformed + misalignment_penalty

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward_transformed,
        'misalignment_penalty': misalignment_penalty,
    }
    
    return total_reward, reward_components
