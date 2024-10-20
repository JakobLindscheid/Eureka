@torch.jit.script
def compute_reward(root_states: torch.Tensor, velocity_scale: float, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Forward velocity component of the reward
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Assuming positive x-direction is forward
    forward_velocity_reward = forward_velocity * velocity_scale

    # Stability penalty based on torso height; encourage staying upright
    torso_height = root_states[:, 2]
    height_penalty = torch.clamp_min(0.5 - torso_height, 0.0)  # Penalty for falling below a certain height

    # Penalize excessive deviation from target to maintain focusing on the track
    dt_reward = -dt  # reward the agent for being fast, create a negative cost for time

    # Calculate the total reward
    total_reward = forward_velocity_reward - height_penalty + dt_reward

    # Applying an exponential transformation for the forward velocity reward
    forward_velocity_temperature = 0.1
    stabilized_forward_velocity_reward = torch.exp(forward_velocity_reward / forward_velocity_temperature) - 1

    # Compile the total reward with transformations
    total_reward = stabilized_forward_velocity_reward - height_penalty + dt_reward

    # Reward components for logging or analysis
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward,
        "height_penalty": height_penalty,
        "dt_reward": dt_reward
    }

    return total_reward, reward_components
