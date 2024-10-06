@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Get the torso position and velocity
    torso_position = root_states[:, 0:3]  # Shape: (N, 3)
    velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Compute the distance to the target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Shape: (N,)
    
    # Compute the speed of the humanoid
    speed = torch.norm(velocity, p=2, dim=-1)  # Shape: (N,)
    
    # Define temperature for reward transformation
    speed_temp = 0.1  # Temperature for speed reward component
    
    # Calculate rewards
    speed_reward = torch.exp(speed / speed_temp)  # Exponentially increase reward with speed
    
    # Negative distance as we want to minimize it
    distance_reward = -torch.exp(distance_to_target / speed_temp)  # Exponentially penalize distance to target
    
    # Total reward is the sum of speed reward and distance reward
    total_reward = speed_reward + distance_reward
    
    # Create a reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'distance_reward': distance_reward,
    }

    return total_reward, reward_components
