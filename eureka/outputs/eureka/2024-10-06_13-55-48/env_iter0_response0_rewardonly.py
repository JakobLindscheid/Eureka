@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso velocity from root states
    torso_velocity = root_states[:, 7:10]  # Assuming this corresponds to the velocity dimensions

    # Calculate the speed (magnitude of velocity)
    speed = torch.norm(torso_velocity, p=2, dim=-1)

    # Define reward components
    speed_reward = speed / dt  # Normalized speed reward

    # Temperature for the reward transformation
    temp_speed = 1.0
    transformed_speed_reward = torch.exp(speed_reward / temp_speed)

    # Total reward
    total_reward = transformed_speed_reward

    # Construct reward components dictionary
    reward_components = {
        "speed_reward": transformed_speed_reward,
    }

    return total_reward, reward_components
