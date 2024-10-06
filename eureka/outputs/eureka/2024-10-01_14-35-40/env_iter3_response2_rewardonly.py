@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # x, y, z velocities extracted
    forward_velocity = velocity[:, 0]  # Forward direction (x axis)

    # Define temperature parameters for normalization
    temperature_running = 0.1
    temperature_action_penalty = 0.1

    # Normalize running reward with temperature
    running_reward = torch.exp(forward_velocity / temperature_running) - 1.0  # Ensures non-negative reward
    
    # Scaled softmax-based action penalty
    action_penalty = -torch.softmax(torch.norm(actions, p=2, dim=-1) / temperature_action_penalty, dim=0)

    # Combined total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
