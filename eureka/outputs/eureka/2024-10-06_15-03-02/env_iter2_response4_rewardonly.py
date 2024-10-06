@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso velocity from the root state
    torso_velocity = root_states[:, 7:10]  # Assuming the velocity is at index 7 to 9
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Compute the speed (magnitude of the velocity)

    # Reward parameters
    speed_temperature = 0.1

    # Compute the speed reward and normalize it using an exponential function
    speed_reward = torch.exp(speed_temperature * speed)

    # Total reward is the speed reward
    total_reward = speed_reward

    # Create a dictionary to store individual components
    reward_components = {
        'speed_reward': speed_reward,
    }

    return total_reward, reward_components
