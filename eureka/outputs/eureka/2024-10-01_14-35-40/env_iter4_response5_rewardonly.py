@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward movement is along the x-axis
    
    # Temperature parameters for scaling
    running_temp = 0.2  # Increased sensitivity for running reward
    action_temp = 0.05  # Keep action penalty temperature low

    # Calculate normalized running reward
    running_reward = torch.exp(forward_velocity ** 2 * running_temp) - 1  # Use square for sensitivity

    # Calculate action penalty based on squared action norms
    action_penalty = -0.02 * (torch.norm(actions, p=2, dim=-1) ** 2)  # Reduced penalty factor

    # Combined total reward 
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
