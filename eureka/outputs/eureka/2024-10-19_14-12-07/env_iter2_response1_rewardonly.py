@torch.jit.script
def compute_reward(velocities: torch.Tensor, target_velocity: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature
    temp_velocity = 0.1

    # Calculate the reward based on the magnitude of the velocity
    speed = torch.norm(velocities, p=2, dim=-1)
    
    # Determine the reward as the difference from the target velocity
    reward_speed = speed - target_velocity

    # Normalize the reward to a fixed range using an exponential function
    reward_normalized_speed = torch.exp(reward_speed / temp_velocity)

    # Total reward is just the normalized speed
    total_reward = reward_normalized_speed

    # Create a dictionary with individual reward components
    reward_components = {
        'speed_reward': reward_normalized_speed
    }

    return total_reward, reward_components
