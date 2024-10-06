@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis
    
    # Temperature parameters for transformations
    running_reward_temp = 0.5
    action_penalty_temp = 0.05
  
    # Reward for running forward: higher speed yields a higher reward
    running_reward = torch.exp(forward_velocity * running_reward_temp)
  
    # Penalty for excessive actions (to encourage efficient movement)
    action_penalty = -torch.exp(torch.norm(actions, p=2, dim=-1) ** 2 * action_penalty_temp)  # Squared penalty for action magnitude
    
    # Combine the rewards
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
