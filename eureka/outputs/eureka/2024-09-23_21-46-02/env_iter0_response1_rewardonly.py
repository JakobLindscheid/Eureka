@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor, target_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define parameters
    temperature_position = 0.5
    temperature_velocity = 0.5

    # Calculate the distance to the target position
    distance = torch.norm(root_positions - target_pos, dim=-1)
    
    # Define the position reward based on the distance to the target
    position_reward = -distance  # Reward is higher (less negative) when closer to the target
    position_reward_transformed = torch.exp(temperature_position * position_reward)
    
    # Calculate the speed (magnitude of linear velocity)
    speed = torch.norm(root_linvels, dim=-1)
    
    # Define a velocity reward where lower speeds are better to encourage hovering
    velocity_reward = -speed  # Reward is higher (less negative) when speed is lower
    velocity_reward_transformed = torch.exp(temperature_velocity * velocity_reward)
    
    # Combine the individual rewards
    total_reward = position_reward_transformed + velocity_reward_transformed

    # Create a dictionary of individual reward components
    reward_components = {
        'position_reward': position_reward_transformed,
        'velocity_reward': velocity_reward_transformed
    }
    
    return total_reward, reward_components
