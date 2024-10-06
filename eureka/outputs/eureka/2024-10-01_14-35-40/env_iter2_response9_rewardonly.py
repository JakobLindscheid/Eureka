@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis
    
    # Set temperature for running reward
    temperature_running = 0.1
    # Reward for running forward: higher speed yields a higher reward, normalized by temperature
    running_reward = torch.exp(forward_velocity / temperature_running)  # Exponentially scale the forward velocity
    
    # Set temperature for action penalty
    temperature_action = 0.5
    # Penalty for excessive actions (to encourage efficient movement), normalized by temperature
    action_penalty = -torch.exp(torch.norm(actions, p=2, dim=-1) / temperature_action)  # Exponential penalty for action magnitude
    
    # Total reward combines both positive and negative components
    total_reward = running_reward + action_penalty
  
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
