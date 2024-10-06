@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transformed reward components
    speed_temp = 5.0
    alignment_temp = 5.0
    
    # Extract the torso position and velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate the forward velocity component (assuming forward is along the x-axis)
    forward_velocity = velocity[:, 0]
    
    # Normalized speed reward (mean subtraction and scaling)
    speed_reward = (forward_velocity - 1.0) / torch.clamp(torch.std(forward_velocity) + 1e-6, min=1e-6)
    speed_reward = torch.exp(speed_reward / speed_temp)

    # Calculate the direction towards the target (assuming the target is on the x-axis as well)
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0  # Ignore the z-axis for forward movement
    direction_to_target_normalized = direction_to_target / (torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6)

    # Calculate the alignment of the torso velocity with the target direction, normalize and scale
    alignment_reward = torch.sum(velocity * direction_to_target_normalized, dim=-1)
    alignment_reward = (alignment_reward - 1.0) / torch.clamp(torch.std(alignment_reward) + 1e-6, min=1e-6)
    alignment_reward = torch.exp(alignment_reward / alignment_temp)

    # Total reward is the combination of modified speed and alignment rewards
    total_reward = speed_reward + alignment_reward

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
    }
    
    return total_reward, reward_components
