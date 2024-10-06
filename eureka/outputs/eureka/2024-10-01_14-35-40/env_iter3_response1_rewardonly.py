@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities
    forward_velocity = velocity[:, 0]  # Forward direction

    # Parameters for temperature scaling
    running_temp = 0.5
    action_temp = 0.1

    # Running reward with exponential transformation
    running_reward = torch.exp(forward_velocity / running_temp)

    # Action penalty with reduced intensity via an exponential transformation
    action_penalty = -0.2 * (torch.norm(actions, p=2, dim=-1) ** 2) / action_temp  # Reduced scale for action penalties

    # Total reward calculation
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
