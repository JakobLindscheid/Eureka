@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis

    # Define temperature parameters for transformations
    running_temp = 0.1  # Adjust this parameter's value to tune the sensitivity of the reward
    action_temp = 0.05  # Adjust this parameter's value to tune the penalty perception

    # Normalize forward velocity to be non-negative and transformed with temperature
    running_reward = torch.exp(forward_velocity / running_temp) - 1  # Ensures reward is positive

    # Action penalty with transformation
    action_penalty = -torch.exp(torch.norm(actions, p=2, dim=-1) / action_temp)

    # Combined total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
