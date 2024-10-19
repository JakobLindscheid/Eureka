@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary components from root_states
    velocity = root_states[:, 7:10]  # Get the velocity of the torso
    forward_velocity = velocity[:, 0]  # Only take the forward (x) component of the velocity

    # Reward components
    speed_reward = torch.square(forward_velocity)  # Amplified reward: square of forward speed
    action_efficiency_reward = 1 - (torch.norm(actions, p=1, dim=-1) / actions.shape[1])  # Reward efficiency of actions

    # Temperature parameters (for reward normalization)
    speed_temp = 0.1  # Temperature for speed reward
    efficiency_temp = 0.05  # Temperature for action efficiency reward

    # Transformed rewards
    transformed_speed_reward = torch.exp(speed_temp * speed_reward)
    transformed_efficiency_reward = torch.exp(efficiency_temp * action_efficiency_reward)

    # Total reward
    total_reward = transformed_speed_reward + transformed_efficiency_reward

    # Create dictionary of individual components
    reward_components = {
        'speed_reward': transformed_speed_reward,
        'action_efficiency_reward': transformed_efficiency_reward
    }

    return total_reward, reward_components
