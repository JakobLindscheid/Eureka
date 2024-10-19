@torch.jit.script
def compute_reward(velocity: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature variables for normalization
    temp_velocity = 0.1
    temp_action = 0.05

    # Reward components
    forward_velocity = velocity[:, 0]  # assuming the forward direction is the first component of velocity
    action_norm = torch.norm(action, p=2, dim=1)  # L2 norm of actions

    # Rewards calculation
    reward_velocity = torch.exp(forward_velocity / temp_velocity)  # Reward for forward velocity
    reward_action = -torch.exp(-action_norm / temp_action)  # Negative reward for high action magnitude (encouraging efficient actions)

    # Total reward
    total_reward = reward_velocity + reward_action

    # Create a dictionary of individual reward components
    reward_components = {
        'forward_velocity_reward': reward_velocity,
        'action_penalty_reward': reward_action
    }

    return total_reward, reward_components
