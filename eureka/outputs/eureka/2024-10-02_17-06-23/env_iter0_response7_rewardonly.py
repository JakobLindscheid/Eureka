@torch.jit.script
def compute_reward(up_vec: torch.Tensor, velocity: torch.Tensor, heading_vec: torch.Tensor, 
                   goal_velocity: float, max_velocity: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Define temperature for reward components
    velocity_temp: float = 0.1
    direction_temp: float = 0.1
    
    # Reward for moving forward based on velocity
    forward_velocity = torch.dot(velocity, heading_vec)  # Projecting velocity onto the heading direction
    velocity_reward = torch.exp(velocity_temp * torch.clamp((forward_velocity - goal_velocity) / max_velocity, min=0, max=1))

    # Reward for maintaining appropriate orientation in relation to the ground
    up_angle_reward = torch.exp(direction_temp * torch.clamp(torch.sum(up_vec * torch.tensor([0.0, 0.0, 1.0], device=up_vec.device)), min=0, max=1))

    # Total reward
    total_reward = velocity_reward + up_angle_reward
    
    # Individual reward components
    reward_components = {
        'velocity_reward': velocity_reward,
        'up_angle_reward': up_angle_reward
    }

    return total_reward, reward_components
