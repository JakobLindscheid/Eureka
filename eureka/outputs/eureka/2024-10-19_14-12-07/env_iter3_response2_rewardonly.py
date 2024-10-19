@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary components from root_states
    velocity = root_states[:, 7:10]  # Get the velocity of the torso
    forward_velocity = velocity[:, 0]  # Take forward (x) component of the velocity

    # Reward components
    speed_reward = forward_velocity * 2.0  # Scale the speed reward for better optimization
    action_penalty = -0.5 * torch.norm(actions, p=1, dim=-1)  # Stronger penalty for actions

    # Temperature parameters for normalization
    speed_temp = 0.05  # Lower temperature to maintain higher variance
    action_temp = 0.1  # Temperature for action penalty

    # Transformed rewards
    transformed_speed_reward = torch.exp(speed_temp * speed_reward)
    transformed_action_penalty = torch.exp(action_temp * action_penalty)

    # Total reward
    total_reward = transformed_speed_reward + transformed_action_penalty

    # Create dictionary of individual components
    reward_components = {
        'speed_reward': transformed_speed_reward,
        'action_penalty': transformed_action_penalty
    }

    return total_reward, reward_components
