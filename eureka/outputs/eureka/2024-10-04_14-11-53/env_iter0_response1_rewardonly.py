@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    temp_velocity = 0.1
    temp_distance = 0.1

    # Calculate forward velocity (x-component of the velocity)
    forward_velocity = velocity[:, 0]  # x direction
    reward_velocity = torch.exp(forward_velocity / temp_velocity)

    # Calculate distance to target along the x-axis (ignoring y and z for this task)
    distance_to_target = target_position[:, 0] - torso_position[:, 0]  # x component
    reward_distance = torch.exp(-torch.abs(distance_to_target) / temp_distance)

    # Combine rewards
    total_reward = reward_velocity + reward_distance

    # Create a dictionary for individual reward components
    reward_components = {
        'forward_velocity_reward': reward_velocity,
        'distance_to_target_reward': reward_distance
    }

    return total_reward, reward_components
