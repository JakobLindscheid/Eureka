@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis

    # Define temperature parameters for scaling
    running_temp = 0.1
    action_temp = 0.05

    # Normalize forward velocity using exponential scaling
    running_reward = torch.exp(forward_velocity / running_temp) - 1.0  # Shifted to ensure non-negativity

    # Penalty for excessive actions with some scaling
    action_penalty = -1 * torch.exp(torch.norm(actions, p=2, dim=-1) / action_temp)  # Allows for a more significant penalty for larger actions

    # Combined total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }

    return total_reward, reward_components
