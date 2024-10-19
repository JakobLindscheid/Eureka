@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary components from root_states
    velocity = root_states[:, 7:10]  # Get the velocity of the torso
    forward_velocity = velocity[:, 0]  # Only take the forward (x) component of the velocity
    previous_velocity = forward_velocity.detach()  # Assuming we can track the previous velocity for acceleration

    # Reward components
    speed_reward = forward_velocity  # Reward proportional to forward speed
    acceleration = forward_velocity - previous_velocity  # Calculate acceleration
    acceleration_reward = torch.clamp(acceleration, min=0)  # Reward only positive acceleration

    action_penalty = -0.5 * torch.norm(actions, p=1, dim=-1)  # Reduced penalty for actions to encourage exploration

    # Temperature parameters (for reward normalization)
    speed_temp = 0.1  # Temperature for speed reward
    acceleration_temp = 0.1  # Temperature for acceleration reward
    action_temp = 0.1  # Temperature for action penalty

    # Transformed rewards
    transformed_speed_reward = torch.exp(speed_temp * speed_reward)
    transformed_acceleration_reward = torch.exp(acceleration_temp * acceleration_reward)
    transformed_action_penalty = torch.exp(action_temp * action_penalty)

    # Total reward
    total_reward = transformed_speed_reward + transformed_acceleration_reward + transformed_action_penalty

    # Create a dictionary of individual components
    reward_components = {
        'speed_reward': transformed_speed_reward,
        'acceleration_reward': transformed_acceleration_reward,
        'action_penalty': transformed_action_penalty
    }

    return total_reward, reward_components
