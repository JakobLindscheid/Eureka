@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward movement is along the x-axis
    
    # Temperature parameters for scaling
    running_temp = 0.5  # Increased responsiveness
    action_temp = 0.2   # Greater sensitivity to action magnitude
    task_temp = 0.05    # Lower impact as compared to running_reward

    # Calculate normalized running reward with new scaling
    running_reward = torch.exp(forward_velocity * running_temp) - 1  # Exponential to encourage forward movement

    # Redesigned action penalty with increased sensitivity to action magnitude
    action_penalty = -0.05 * (torch.norm(actions, p=2, dim=-1) ** 2)  # Squared norm for harsher penalty

    # Introducing task_score similarly but with a lower temperature to diminish its direct influence
    task_score = torch.exp(running_reward * task_temp) - 1  # Consider it based on running performance

    # Combined total reward
    total_reward = running_reward + action_penalty + task_score

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty,
        "task_score": task_score,
    }
    
    return total_reward, reward_components
