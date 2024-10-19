@torch.jit.script
def compute_reward(root_state: torch.Tensor, target_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Computes the reward for the Ant3 to encourage forward running."""
    # Constants
    speed_weight = 1.0  # Weight for speed in reward
    reaching_target_weight = 10.0  # Weight for reaching the target

    # Speed calculation
    torso_velocity = root_state[:, 7:10]  # Extract the linear velocity
    forward_speed = torso_velocity[:, 0]  # Assuming the forward direction is the x-axis
    reward_speed = forward_speed * speed_weight  # Incentivize forward speed
    
    # Distance to the target
    torso_position = root_state[:, 0:3]  # Extract the torso position
    distance_to_target = torch.norm(target_pos - torso_position, p=2, dim=-1)  # Calculate distance to target
    reward_reaching_target = -distance_to_target * reaching_target_weight  # Incentivize reaching the target

    # Total reward
    total_reward = reward_speed + reward_reaching_target
    
    # Transformations
    temp_speed = 1.0  # Temperature parameter for speed reward transformation
    temp_target = 1.0  # Temperature parameter for target reward transformation
    
    transformed_reward_speed = torch.exp(reward_speed / temp_speed)  # Apply transformation for speed
    transformed_reward_reaching_target = torch.exp(reward_reaching_target / temp_target)  # Apply transformation for target reaching

    # Total transformed reward
    transformed_total_reward = transformed_reward_speed + transformed_reward_reaching_target
    
    # Creating a dictionary for individual components
    reward_components = {
        'speed_reward': transformed_reward_speed,
        'target_reward': transformed_reward_reaching_target
    }

    return transformed_total_reward, reward_components
