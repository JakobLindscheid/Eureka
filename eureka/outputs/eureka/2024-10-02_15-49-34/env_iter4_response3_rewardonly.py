@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso position and velocity
    torso_position = root_states[:, 0:3]  # Shape: (batch_size, 3)
    velocity = root_states[:, 7:10]  # Shape: (batch_size, 3)

    # Define the target position (goal in x-direction)
    goal_position = targets  # Targets are predefined in the environment

    # Reward for forward speed (x component of velocity)
    forward_speed_reward = velocity[:, 0]  # Reward based on forward x speed
    forward_speed_reward = forward_speed_reward * 2.0  # Scale the forward speed reward

    # Calculate progress towards the target in the x-direction
    progress_towards_target = torch.clamp(goal_position[:, 0] - torso_position[:, 0], min=0.0)  # Reward progress towards the target

    # Scale progress significantly
    progress_reward = progress_towards_target * 5.0  # Emphasize reaching the goal

    # Combine rewards
    total_reward = forward_speed_reward + progress_reward

    # Temperature parameters for both components
    temp_forward_speed = 1.0
    temp_progress = 1.0
    
    # Apply temperature scaling for both components
    forward_speed_reward_transformed = torch.exp(forward_speed_reward / temp_forward_speed)
    progress_reward_transformed = torch.exp(progress_reward / temp_progress)

    # Total reward after transformation
    total_reward_transformed = forward_speed_reward_transformed + progress_reward_transformed
    
    # Create reward components dictionary
    reward_components = {
        'forward_speed_reward': forward_speed_reward,
        'progress_reward': progress_reward
    }
    
    return total_reward_transformed, reward_components
