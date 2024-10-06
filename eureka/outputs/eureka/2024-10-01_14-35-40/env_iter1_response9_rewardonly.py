@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis
    
    # Reward for running forward: positive reward for higher speed and sustained effort
    velocity_scale = 10.0  # Scale for forward velocity reward
    running_reward = torch.exp(forward_velocity / velocity_scale)  # Adjusted temperature for sensitivity
    running_reward = torch.clip(running_reward, min=0.0, max=1.0)  # Normalize to [0, 1]
    
    # Increased penalty for excessive actions (to discourage erratic movements)
    action_penalty_scale = 5.0  # Increased scale for action penalty
    action_penalty = -torch.norm(actions, p=2, dim=-1)  # Penalty based on action magnitude
    action_penalty = torch.exp(action_penalty / action_penalty_scale)  # Adjusted temperature for penalty
    action_penalty = -torch.clip(action_penalty, max=0.0, min=-1.0)  # Ensuring penalty is in a meaningful range
    
    # Calculate total reward
    total_reward = running_reward + action_penalty
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
