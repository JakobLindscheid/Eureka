@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis

    # Define temperature for normalization
    running_temp = 0.5
    action_temp = 0.2

    # Normalize running reward to have a higher emphasis on forward movement
    running_reward = torch.exp(forward_velocity / running_temp) - 1.0  # Exponential reward based on forward speed

    # Reduce the action penalty's impact while still keeping accountability
    action_penalty = -0.1 * (torch.norm(actions, p=2, dim=-1) ** 2) / action_temp  # Scale down penalty significantly 

    # Combined total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
