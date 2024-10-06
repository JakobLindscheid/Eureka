@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Root states structure:
    # root_states[:, 0:3] - position (x, y, z)
    # root_states[:, 7:10] - velocity (vx, vy, vz)
    
    # Constants
    speed_weight: float = 1.0
    
    # Extract the forward speed (x-component of the velocity)
    forward_speed = root_states[:, 7]  # Assuming index 7 corresponds to vx

    # Compute the reward as the forward speed
    speed_reward = forward_speed * speed_weight  # Direct reward for forward speed
    
    # Normalization (using temperature)
    temp_speed: float = 0.1
    normalized_speed_reward = torch.exp(speed_reward / temp_speed)

    total_reward = normalized_speed_reward
    
    return total_reward, {'speed_reward': normalized_speed_reward}
