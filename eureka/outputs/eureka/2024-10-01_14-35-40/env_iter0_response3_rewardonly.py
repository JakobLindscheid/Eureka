@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor,
                   actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis
    
    # Reward for running forward: positive reward for higher speed
    running_reward = torch.exp(forward_velocity / 10.0)  # Temperature can be adjusted
    running_reward = torch.clip(running_reward, min=0.0, max=1.0)  # Normalize to [0, 1]
    
    # Penalty for excessive actions (to discourage too erratic movements)
    action_penalty = -torch.norm(actions, p=2, dim=-1)  # Penalty based on action magnitude
    action_penalty = torch.exp(action_penalty / 5.0)  # Temperature for penalty
    
    # Calculate total reward
    total_reward = running_reward + action_penalty
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
