@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor, potentials: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the velocity, which is part of the root states
    velocity = root_states[:, 7:10]
    
    # Calculate the speed as the norm of the velocity vector
    speed = torch.norm(velocity, p=2, dim=-1)

    # Set temperature variable to normalize the rewards
    temperature_speed = 0.1

    # Reward for speed (we apply an exponential transformation for normalization)
    speed_reward = torch.exp(temperature_speed * speed)
    
    # Reward for distance covered based on potential change
    distance_reward = potentials - prev_potentials
    
    # Set temperature variable for distance reward
    temperature_distance = 0.1
    
    # Transform distance reward
    distance_reward_transformed = torch.exp(temperature_distance * distance_reward)

    # Total reward as a combination of speed and distance reward
    total_reward = speed_reward + distance_reward_transformed

    # Individual reward components
    individual_rewards = {
        'speed_reward': speed_reward,
        'distance_reward': distance_reward_transformed
    }

    return total_reward, individual_rewards
