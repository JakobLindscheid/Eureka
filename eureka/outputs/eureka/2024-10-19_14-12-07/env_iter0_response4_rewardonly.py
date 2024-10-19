@torch.jit.script
def compute_reward(torso_velocity: torch.Tensor, prev_potential: torch.Tensor, potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature for normalizing reward components
    speed_temp = 0.1
    potential_temp = 0.1
    
    # Compute speed of the humanoid (magnitude of torso velocity)
    speed = torch.norm(torso_velocity, p=2, dim=-1)
    
    # Calculate the reward based on speed
    speed_reward = speed / speed_temp
    # Normalize the speed reward
    speed_reward = torch.exp(speed_reward)  # Exponential transformation to encourage higher speeds

    # Calculate reward based on change in potential
    potential_change = potentials - prev_potential
    potential_reward = potential_change / potential_temp
    potential_reward = torch.exp(potential_reward)  # Exponential transformation to reward increases in potential
    
    # Total reward is a weighted sum of speed and potential rewards
    total_reward = speed_reward + potential_reward

    # Create the reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'potential_reward': potential_reward
    }
    
    return total_reward, reward_components
