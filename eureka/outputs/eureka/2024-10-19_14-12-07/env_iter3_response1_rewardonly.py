@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary components from root_states
    velocity = root_states[:, 7:10]  # Get the velocity of the torso
    forward_velocity = velocity[:, 0]  # Only take the forward (x) component of the velocity

    # Reward components
    speed_reward = forward_velocity  # Reward proportional to forward speed

    # Scale actions to limit penalty; using a norm for more precise control
    # Instead of simply penalizing, we'll also reward less movement
    action_penalty = -torch.norm(actions, p=2, dim=-1)  # Penalty for large action magnitudes (L2 norm)

    # Temperature parameters (for reward normalization)
    speed_temp = 0.05  # Reduced temperature for speed reward
    action_temp = 0.02  # Reduced temperature for action penalty

    # Transformed rewards to ensure scaling
    transformed_speed_reward = torch.exp(speed_temp * speed_reward) - 1  # Shift to keep positive
    transformed_action_penalty = torch.exp(action_temp * action_penalty)

    # Total reward combining speed reward positively and action penalty as a negative contribution
    total_reward = transformed_speed_reward - transformed_action_penalty

    # Create dictionary of individual components
    reward_components = {
        'speed_reward': transformed_speed_reward,
        'action_penalty': transformed_action_penalty
    }

    return total_reward, reward_components
