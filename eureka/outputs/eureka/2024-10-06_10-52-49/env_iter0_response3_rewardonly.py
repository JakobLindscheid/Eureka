@torch.jit.script
def compute_reward(velocity: torch.Tensor, target_speed: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameter for normalization
    temp_velocity = 0.1 # Feel free to adjust this value

    # Compute reward based on the speed of the humanoid
    current_speed = torch.norm(velocity, p=2, dim=-1)  # Euclidean norm of the velocity
    speed_reward = current_speed - target_speed  # Reward for exceeding the target speed
    
    # Normalize speed reward
    normalized_speed_reward = torch.exp(speed_reward * temp_velocity)

    # Total reward is the normalized speed reward
    total_reward = normalized_speed_reward

    # Create a dictionary for individual reward components
    reward_components = {
        "speed_reward": normalized_speed_reward,
    }

    return total_reward, reward_components
