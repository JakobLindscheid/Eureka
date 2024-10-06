@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 5.0  # Increased to reduce the impact of extreme values
    alignment_temp = 5.0  # Increased to allow better scale control
    position_temp = 10.0  # New temperature for positional reward scaling

    # Extract the torso position and velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate the forward velocity component (assuming forward is along the x-axis)
    forward_velocity = velocity[:, 0]
    
    # Reward component for moving forward quickly with log scaling
    speed_reward = torch.log(torch.abs(forward_velocity) + 1e-6) / speed_temp

    # Calculate the direction towards the target (ignoring the z-axis)
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0
    direction_to_target_normalized = direction_to_target / (torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6)

    # Calculate alignment reward based on how well velocity matches the target direction
    alignment_reward = torch.log(torch.relu(torch.sum(velocity * direction_to_target_normalized, dim=-1)) + 1e-6) / alignment_temp

    # New positional reward based on how close the torso's x position is to the target's x position
    position_reward = -torch.abs(torso_position[:, 0] - targets[:, 0]) / position_temp

    # Total reward is the combination of all components
    total_reward = speed_reward + alignment_reward + position_reward

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'position_reward': position_reward,
    }
    
    return total_reward, reward_components
