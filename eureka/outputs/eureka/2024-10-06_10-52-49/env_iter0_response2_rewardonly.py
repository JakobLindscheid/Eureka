@torch.jit.script
def compute_reward(root_states: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso velocity from the root states
    velocity = root_states[:, 7:10]  # Assuming velocity is stored in this slice
    
    # Compute the linear speed of the humanoid (magnitude of the velocity vector)
    speed = torch.norm(velocity, p=2, dim=1)
    
    # Reward for running faster
    reward_speed = speed
    
    # Define the temperature parameter for normalization
    temp_speed = 1.0  # Temperature value for speed normalization
    normalized_reward_speed = torch.exp(reward_speed / temp_speed)

    # Total reward is simply the normalized reward for speed
    total_reward = normalized_reward_speed

    # Create a dictionary for the individual reward components
    reward_components = {
        'speed_reward': normalized_reward_speed
    }

    return total_reward, reward_components
