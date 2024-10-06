@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso's velocity in the x direction (forward velocity)
    velocity = root_states[:, 7:10]  # Assuming the order is [x, y, z]
    forward_velocity = velocity[:, 0]  # Forward movement is along the x-axis
     
    # Temperature parameters for scaling to improve learning
    running_temp = 0.5
    action_temp = 0.1
    task_temp = 0.3
    
    # Calculate normalized running reward
    running_reward = torch.exp(forward_velocity * running_temp) - 1  # Adjusted scaling for better variance
    
    # Calculate a more aggressive action penalty with scaling
    action_penalty = -0.5 * (torch.norm(actions, p=2, dim=-1) ** 2) * action_temp  # Enhanced penalty factor
    
    # Task score - we can modify its contribution but not drastically replace it
    task_reward = torch.exp(forward_velocity * task_temp)  # Scaled based on forward velocity
    
    # Combined total reward
    total_reward = running_reward + action_penalty + task_reward

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty,
        "task_reward": task_reward
    }
    
    return total_reward, reward_components
