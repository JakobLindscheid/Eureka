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
    reward_forward_speed = forward_velocity  # Reward based on forward speed
    sustained_speed = torch.max(forward_velocity, torch.tensor(0.0, device=root_states.device))  # Only positive forward velocity
    heading_dot_product = torch.sum(torso_velocity * (to_target / (distance_to_target.unsqueeze(-1) + 1e-5)), dim=-1)  # Directional alignment

    # Adjust the temperature parameters for transformations
    temperature_forward = 5.0   # Sensitivity for speed
    temperature_sustained_speed = 10.0  # Scaling down to lessen dominance
    temperature_heading = 5.0    # Optimizing heading effectiveness

    # Transforming the rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_forward) - 1
    transformed_sustained_speed = torch.exp(sustained_speed / temperature_sustained_speed) - 1  
    transformed_reward_heading = torch.exp(heading_dot_product / temperature_heading)  # Exponential for the heading

    # Introduce a linearly scaled measure for task score correlated with distance minimized
    task_score = (torch.max(torch.tensor(0.0, device=root_states.device), 1 - (distance_to_target / 10)) 
                  * torch.exp((reward_heading + reward_forward_speed) / 5))  # Scale to the total progress
    transformed_task_score = torch.exp(task_score) - 1  # Normalize

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_sustained_speed + transformed_reward_heading + transformed_task_score

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'sustained_speed': transformed_sustained_speed,
        'reward_heading': transformed_reward_heading,
        'task_score': transformed_task_score
    }

    return total_reward, reward_components
