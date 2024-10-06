@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso velocity
    torso_velocity = root_states[:, 7:10]  # Assuming the velocity is in the 7th, 8th, and 9th index
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Calculate speed as the L2 norm of the velocity vector

    # Define a temperature parameter for transforming the reward
    temperature_speed = 1.0
    normalized_speed_reward = torch.exp(speed / temperature_speed)  # Transform speed reward for normalization

    # Total reward is based on the normalized speed
    total_reward = normalized_speed_reward

    # Reward components dictionary
    reward_components = {
        "speed": normalized_speed_reward
    }

    return total_reward, reward_components
