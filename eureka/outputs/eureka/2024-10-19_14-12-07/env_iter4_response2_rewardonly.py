@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary components from root_states
    velocity = root_states[:, 7:10]  # Torso velocity
    speed = torch.norm(velocity, p=2, dim=-1)  # Overall speed
    forward_velocity = velocity[:, 0]  # Forward (x) component

    # Reward components
    speed_reward = torch.exp(0.1 * speed)  # Exponential scaling of speed reward
    direction_bonus = torch.exp(0.2 * torch.clamp(forward_velocity, min=0))  # Stronger positive reward for forward movement
    action_penalty = -torch.norm(actions, p=2, dim=-1) ** 2  # Squared action penalty

    # Temperature parameters for normalization
    speed_temp = 0.05  # Speed reward temperature
    direction_temp = 0.1  # Direction bonus temperature
    action_temp = 0.1  # Action penalty temperature

    # Transformed rewards
    transformed_speed_reward = speed_reward * speed_temp  # Normalize speed reward
    transformed_direction_bonus = direction_bonus * direction_temp  # Normalize direction bonus
    transformed_action_penalty = torch.exp(action_penalty * action_temp)  # Normalize action penalty

    # Total reward
    total_reward = transformed_speed_reward + transformed_direction_bonus + transformed_action_penalty

    # Create dictionary of individual components
    reward_components = {
        'speed_reward': transformed_speed_reward,
        'direction_bonus': transformed_direction_bonus,
        'action_penalty': transformed_action_penalty
    }

    return total_reward, reward_components
