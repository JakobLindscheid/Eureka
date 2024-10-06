@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity (the forward velocity)
    velocity = root_states[:, 7:10]  # Assuming x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Extract the x-component for forward motion

    # Calculate the reward based on the forward velocity
    forward_velocity_reward = forward_velocity

    # Normalize the reward using an exponential function and a temperature parameter
    temperature_forward = 0.1  # This is a constant that can be tuned later
    normalized_reward = torch.exp(forward_velocity_reward / temperature_forward)

    # Total reward is the normalized reward
    total_reward = normalized_reward

    # Create a dictionary to return individual components of the reward
    reward_components = {
        'forward_velocity_reward': forward_velocity_reward,
    }

    return total_reward, reward_components
