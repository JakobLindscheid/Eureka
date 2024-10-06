@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis

    # New temperature parameters for normalizing rewards
    running_temp = 0.1  # A small temperature to soften the impact
    action_temp = 0.05  # Even smaller to diminish the penalty's influence

    # Normalize running reward with temperature; ensure it's non-negative
    running_reward = torch.exp(forward_velocity / running_temp)  # Exponential growth on reward

    # Penalty for excessive actions with normalization; scaling down
    action_penalty = -0.1 * (torch.norm(actions, p=2, dim=-1) ** 2) / action_temp

    # Combined total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
