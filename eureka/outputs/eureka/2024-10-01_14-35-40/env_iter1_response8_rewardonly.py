@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor,
                   actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis

    # Reward for running forward: positive reward for higher average speed
    average_velocity = torch.mean(forward_velocity)
    running_reward = torch.exp(average_velocity / 5.0)  # Encourage average speed
    
    # Aggressive penalty for excessive action usage (encouraging smoother movements)
    action_penalty = -torch.norm(actions, p=2, dim=-1) * 2.0  # Stronger penalty
    action_penalty = torch.exp(action_penalty / 5.0)  # Scaled version

    # New component: reward based on forward progress
    progress_reward = torch.clip(forward_velocity / 2.0, min=0.0, max=1.0)  # Reward based on immediate speed

    # Calculate total reward
    total_reward = running_reward + action_penalty + progress_reward
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty,
        "progress_reward": progress_reward
    }
    
    return total_reward, reward_components
