@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # x, y, z velocities
    forward_velocity = velocity[:, 0]  # Forward movement is along the x-axis
    
    # Calculate forward speed reward
    forward_speed_reward = torch.maximum(forward_velocity, torch.tensor(0.0, device=forward_velocity.device))

    # Temperature parameters for scaling
    forward_speed_temp = 0.2
    action_temp = 0.05

    # Normalize the forward speed reward
    normalized_forward_speed_reward = torch.exp(forward_speed_reward * forward_speed_temp) - 1  # Ensure non-negative

    # Calculate action penalty (incentivizing efficiency in actions)
    action_penalty = -torch.exp(torch.norm(actions, p=2, dim=-1) * action_temp)

    # Combined total reward
    total_reward = normalized_forward_speed_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": normalized_forward_speed_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
