@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity components
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is along the x axis

    # Use temperature variables for reward transformations
    temp_running = 0.1  # Temperature for the running reward
    temp_action_penalty = 0.05  # Temperature for the action penalty

    # Normalize the running reward for forward movement
    running_reward = torch.exp(forward_velocity / temp_running) - 1  # Shifted to make it always positive

    # Update action penalty to be less impactful
    action_penalty = -0.2 * torch.norm(actions, p=2, dim=-1) / temp_action_penalty  # Penalty scaled down
   
    # Combined total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary for monitoring
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
