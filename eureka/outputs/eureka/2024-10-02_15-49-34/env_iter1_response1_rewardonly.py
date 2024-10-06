@torch.jit.script
def compute_reward(
    torso_position: torch.Tensor,
    velocity: torch.Tensor,
    targets: torch.Tensor,
    dt: float,
    max_speed: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    speed_temp = 0.1
    distance_temp = 0.1
    
    # Direction towards target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)

    # Reward for forward velocity (encouraging to run fast)
    forward_velocity = velocity[:, 0]  # Considering forward direction as the first component
    speed_reward = torch.exp(forward_velocity / max_speed) - 1  # Normalized speed reward
    
    # Reward based on distance to target (decrease as it gets closer)
    distance_reward = -torch.exp(distance_to_target / max_speed) + 1  # Negative for getting closer
    
    # Total reward
    total_reward = speed_reward + distance_reward

    # Create a dictionary for individual rewards
    reward_components = {
        'speed_reward': speed_reward,
        'distance_reward': distance_reward,
    }

    return total_reward, reward_components
