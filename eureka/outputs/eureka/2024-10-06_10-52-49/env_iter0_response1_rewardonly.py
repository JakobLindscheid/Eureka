@torch.jit.script
def compute_reward(torso_velocity: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameter for reward scaling
    temp_velocity = 0.1
    temp_action = 0.1

    # Reward component for forward velocity (x-direction)
    forward_velocity = torso_velocity[:, 0]  # assuming the first element corresponds to forward velocity
    velocity_reward = torch.exp(forward_velocity / temp_velocity)

    # Reward component for minimizing action magnitude (encouraging efficient action usage)
    action_magnitude = torch.norm(action, p=2, dim=-1)
    action_penalty = -torch.exp(-action_magnitude / temp_action)

    # Total reward aggregation
    total_reward = velocity_reward + action_penalty

    # Reward components dictionary
    reward_components = {
        'velocity_reward': velocity_reward,
        'action_penalty': action_penalty,
    }

    return total_reward, reward_components
