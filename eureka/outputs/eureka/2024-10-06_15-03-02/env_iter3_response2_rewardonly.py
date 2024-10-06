@torch.jit.script
def compute_reward(root_states: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant information
    torso_position = root_states[:, 0:3]  # Torso position
    torso_velocity = velocity  # Current velocity

    # Define temperature variables
    speed_temp = 0.1

    # Compute speed (magnitude of velocity)
    speed = torch.norm(torso_velocity, p=2, dim=-1)

    # Reward for running faster
    speed_reward = speed * speed_temp
    speed_reward_transformed = torch.exp(speed_reward)

    # Total reward
    total_reward = speed_reward_transformed

    # Individual reward components
    reward_components = {
        "speed_reward": speed_reward_transformed
    }

    return total_reward, reward_components
