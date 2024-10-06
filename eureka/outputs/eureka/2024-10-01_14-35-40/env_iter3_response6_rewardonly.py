@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extracting the forward velocity (x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Temperature parameters for scaling rewards
    temp_running = 0.1
    temp_action_penalty = 0.1

    # Normalized running reward using temperature
    running_reward = torch.exp(forward_velocity / temp_running) - 1.0  # Using exp to ensure positive rewards

    # Penalty for actions, reduced scale with temperature
    action_penalty = -0.5 * torch.norm(actions, p=2, dim=-1) / temp_action_penalty  # Adjusting penalty

    # Total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }

    return total_reward, reward_components
