@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis

    # Reward for running forward: higher speed yields a higher reward
    running_temp = 0.1
    running_reward = torch.exp(forward_velocity / running_temp)  # Exponential scaling to incentivize higher speeds

    # Penalty for excessive actions (to encourage efficient movement)
    action_temp = 0.1
    action_penalty = -torch.exp(torch.norm(actions, p=2, dim=-1) / action_temp)  # Exponential penalty for action magnitude
    
    # Total reward combines running reward and action penalty
    total_reward = torch.clip(running_reward + action_penalty, min=-10.0, max=10.0)

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
