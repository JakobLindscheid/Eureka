@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis
    
    # Enhanced reward for running forward: positive reward for higher speed
    running_reward_temp = 5.0  # Temperature for scaling
    running_reward = torch.exp(forward_velocity / running_reward_temp)
    
    # Normalizing running reward
    running_reward = torch.clip(running_reward, min=0.0, max=1.0)  
    
    # Re-scaled action penalty
    action_penalty_temp = 2.0  # New temperature value for the action penalty
    action_penalty = -torch.norm(actions, p=2, dim=-1)  # Penalty based on action magnitude
    action_penalty = torch.exp(action_penalty / action_penalty_temp)  # Transform and scale
    
    # Calculate total reward, focusing on running reward
    total_reward = running_reward + action_penalty
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
