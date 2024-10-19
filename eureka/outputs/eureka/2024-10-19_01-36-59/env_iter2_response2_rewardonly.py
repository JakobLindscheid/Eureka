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
    sustained_speed = torch.max(forward_velocity, torch.tensor(0.0, device=root_states.device))  # Only positive forward velocity matters
    
    # Reward for heading towards the target
    reward_heading = torch.exp(-distance_to_target / (10.0 + 1e-5))  # Encourage to move closer to target

    # Adjust the temperature parameters for transformations
    temperature_forward = 10.0
    temperature_sustained_speed = 5.0
    temperature_heading = 5.0

    # Transforming the rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_forward) - 1
    transformed_sustained_speed = torch.exp(sustained_speed / temperature_sustained_speed) - 1  
    transformed_reward_heading = reward_heading  # No transformation necessary since it's normalized

    # Total reward calculation
    total_reward = transformed_reward_forward_speed + transformed_sustained_speed + transformed_reward_heading

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'sustained_speed': transformed_sustained_speed,
        'reward_heading': transformed_reward_heading,
    }

    return total_reward, reward_components
