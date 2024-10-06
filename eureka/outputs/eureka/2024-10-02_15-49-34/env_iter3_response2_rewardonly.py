@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for reward normalization
    forward_velocity_temp = 0.1
    position_temp = 0.1

    # Reward for forward velocity
    forward_velocity_reward = velocity[:, 0]  # Assuming the forward direction is the x-axis
    normalized_forward_velocity_reward = torch.exp(forward_velocity_temp * forward_velocity_reward)

    # Reward for proximity to the target goal
    distance_to_target = torch.norm(targets - torso_position, p=2, dim=-1)
    proximity_reward = -distance_to_target  # Closer to target, higher the reward
    normalized_proximity_reward = torch.exp(position_temp * proximity_reward)

    # Total reward as a combination of forward velocity and proximity to target
    total_reward = normalized_forward_velocity_reward + normalized_proximity_reward

    # Dictionary of individual reward components
    reward_components = {
        'forward_velocity_reward': normalized_forward_velocity_reward,
        'proximity_reward': normalized_proximity_reward,
    }

    return total_reward, reward_components
