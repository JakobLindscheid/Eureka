@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor,
                   actions: torch.Tensor, episode_length: float, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                   
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis
    
    # Reward for running forward: exponential reward for higher speed
    running_reward = torch.exp(forward_velocity / 5.0)  # Make it more sensitive to speed changes
    running_reward = torch.clip(running_reward, min=0.0, max=1.0)  # Normalize to [0, 1]
    
    # New action penalty calculated with a smooth soft plus function
    action_penalty = -torch.log(torch.exp(-torch.norm(actions, p=2, dim=-1) / 10.0) + 1.0)  # Smoother penalty structure
    action_penalty = torch.exp(action_penalty / 2.0)  # Temperature can be adjusted for the penalty
    
    # Reward for maintaining longer episode lengths
    length_reward = torch.clamp(episode_length / 1000.0, 0.0, 1.0)  # Normalize episode length to [0, 1]
    
    # Calculate total reward
    total_reward = running_reward + action_penalty + length_reward
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty,
        "length_reward": length_reward
    }
    
    return total_reward, reward_components
