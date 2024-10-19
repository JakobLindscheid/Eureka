@torch.jit.script
def compute_reward(torso_velocity: torch.Tensor, target_direction: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameter for reward normalization
    velocity_temp = 1.0  # Adjust the temperature for scaling the velocity reward

    # Compute the forward speed (velocity along the target direction)
    forward_speed = torch.dot(torso_velocity, target_direction)

    # Reward is the forward speed normalized by the temperature
    reward = torch.exp(forward_speed / velocity_temp)

    # Components for individual rewards
    reward_components = {
        "forward_speed": forward_speed,
    }

    return reward, reward_components
