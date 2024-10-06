@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float, action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the necessary variables
    torso_velocity = root_states[:, 7:10]  # Linear velocity of the torso
    forward_velocity = torso_velocity[:, 0]  # Velocity along the x-axis (forward motion)

    # Define temperature parameters for reward components
    forward_vel_temp = 0.1
    action_penalty_temp = 0.1

    # Compute the rewards
    forward_reward = torch.exp(forward_velocity * forward_vel_temp)  # Positive reward for forward speed
    action_penalty = -torch.exp(torch.norm(action, p=2, dim=-1) * action_penalty_temp)  # Penalty for large actions

    # Total reward
    total_reward = forward_reward + action_penalty

    # Create the reward components dictionary
    reward_components = {
        "forward_reward": forward_reward,
        "action_penalty": action_penalty
    }

    return total_reward, reward_components
