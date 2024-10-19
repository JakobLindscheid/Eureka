@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)
    torso_position = root_states[:, 0:3]
    
    # Calculate distance to the target (2D, ignoring height)
    to_target = targets - torso_position
    distance_to_target = torch.sqrt(to_target[:, 0]**2 + to_target[:, 1]**2)

    # New reward components
    reward_forward_speed = forward_velocity  # Encourages rapid forward motion
    sustained_speed = torch.mean(forward_velocity[forward_velocity > 0])  # Average positive speed, rewards consistent running
    
    # New heading reward based on the direction towards the target
    direction_to_target = to_target / (distance_to_target.unsqueeze(-1) + 1e-5)  # Normalize direction
    heading_dot_product = torch.sum(torso_velocity * direction_to_target, dim=-1)  # Dot product gives alignment

    # Adjust the temperature parameters for transformations
    temperature_forward = 4.0  # Decrease to allow more sensitivity
    temperature_sustained_speed = 5.0  # Increase to reduce saturated values
    temperature_heading = 6.0  # Keep to control heading improvements

    # Transforming the rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_forward) - 1
    transformed_sustained_speed = torch.exp(sustained_speed / temperature_sustained_speed) - 1  
    transformed_reward_heading = torch.exp(heading_dot_product / temperature_heading)  # Exponential for the heading
    normalized_task_score = torch.sigmoid(torch.tensor(torch.mean(reward_forward_speed) + torch.mean(sustained_speed), device=root_states.device))  # Soft normalized task score

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_sustained_speed + transformed_reward_heading + normalized_task_score

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'sustained_speed': transformed_sustained_speed,
        'reward_heading': transformed_reward_heading,
        'task_score': normalized_task_score,
    }

    return total_reward, reward_components
