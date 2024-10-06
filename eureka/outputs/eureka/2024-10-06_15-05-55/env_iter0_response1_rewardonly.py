@torch.jit.script
def compute_reward(velocity: torch.Tensor, heading_vec: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Rewards for the speed of the humanoid
    speed_reward = torch.norm(velocity, p=2, dim=-1)  # Calculate the speed as the norm of the velocity

    # Encourage the humanoid to move in the direction of its heading vector
    forward_reward = torch.sum(velocity * heading_vec, dim=-1)

    # Total reward is the sum of speed and directional rewards
    total_reward = speed_reward + forward_reward

    # Normalize the rewards using an exponential function for stability
    temp_speed = 1.0  # Temperature for speed reward transformation
    temp_forward = 1.0  # Temperature for forward reward transformation

    normalized_speed_reward = torch.exp(total_reward / temp_speed)
    
    # Reward breakdown
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'forward_reward': forward_reward,
    }

    return normalized_speed_reward, reward_components
