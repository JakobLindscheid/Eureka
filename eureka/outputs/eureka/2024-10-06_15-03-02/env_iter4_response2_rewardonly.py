@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the velocity of the torso to compute running speed
    torso_velocity = root_states[:, 7:10]  # Assuming the torso velocity is in the 7-9 indices
    
    # Calculate the speed as the norm of the torso's velocity vector
    speed = torch.norm(torso_velocity, p=2, dim=-1)
    
    # Reward based on speed, we introduce a "temperature" variable
    temperature_speed = 0.1
    speed_reward = torch.exp(temperature_speed * speed) - 1  # Normalize reward in a fitting range
    
    # Combining the reward components
    reward = speed_reward.sum()  # Total reward
    
    # Create a dictionary of individual reward components
    reward_components = {
        'speed_reward': speed_reward
    }
    
    return reward, reward_components
