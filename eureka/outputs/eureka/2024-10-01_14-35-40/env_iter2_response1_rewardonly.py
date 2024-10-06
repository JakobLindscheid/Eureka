@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants for temperature scaling
    temp_running = 0.1  # Temperature for running reward
    temp_action = 0.05  # Temperature for action penalty

    # Extract forward velocity
    velocity = root_states[:, 7:10]  # Extract velocities (x, y, z)
    forward_velocity = velocity[:, 0]  # Forward-facing velocity (x-axis)

    # Calculate running reward: encourage both speed and positive acceleration
    running_reward = forward_velocity + torch.norm(velocity[:, 1:], dim=1)  # Adding lateral speed for better overall movement
    running_reward = torch.exp(temp_running * running_reward) - 1  # Transform with temperature (shifting to range)

    # Calculate action penalty based on squared magnitude of actions
    action_penalty = -torch.norm(actions, p=2, dim=-1) ** 2
    action_penalty = torch.exp(temp_action * action_penalty) - 1  # Transform to normalize

    # Total reward calculation
    total_reward = torch.clip(running_reward + action_penalty, min=-10.0, max=10.0)

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
