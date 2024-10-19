@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary components from root_states
    velocity = root_states[:, 7:10]  # Get the velocity of the torso
    speed = torch.norm(velocity, p=2, dim=-1)  # Magnitude of speed
    forward_velocity = velocity[:, 0]  # Only take the forward (x) component of velocity

    # Compute acceleration
    prev_velocity = torch.tensor([0.0, 0.0, 0.0], device=root_states.device)  # Placeholder for previous velocity
    acceleration = velocity - prev_velocity  # Delta from previous state (assumes proper integration in real context)

    # Reward components
    speed_reward = speed  # Reward proportional to overall speed
    acceleration_reward = torch.norm(acceleration, p=2, dim=-1)  # Reward for acceleration
    direction_bonus = torch.clamp(forward_velocity / (1e-6 + speed), min=0)  # Normalize forward velocity
    action_penalty = -torch.norm(actions, p=2, dim=-1) * 0.1  # Penalize excessive action usage

    # Temperature parameters (for reward normalization)
    speed_temp = 0.02  # Temperature for speed reward
    acceleration_temp = 0.01  # Temperature for acceleration reward
    direction_temp = 0.15  # Adjusted temperature for direction bonus
    action_temp = 0.05  # Temperature for action penalty

    # Transformed rewards
    transformed_speed_reward = torch.exp(speed_temp * speed_reward)  # Normalizing speed reward
    transformed_acceleration_reward = torch.exp(acceleration_temp * acceleration_reward)  # Normalizing acceleration reward
    transformed_direction_bonus = torch.exp(direction_temp * direction_bonus)  # Normalizing direction bonus
    transformed_action_penalty = torch.exp(action_temp * action_penalty)  # Normalizing action penalty

    # Total reward
    total_reward = (transformed_speed_reward + transformed_acceleration_reward + 
                    transformed_direction_bonus + transformed_action_penalty)

    # Create dictionary of individual components
    reward_components = {
        'speed_reward': transformed_speed_reward,
        'acceleration_reward': transformed_acceleration_reward,
        'direction_bonus': transformed_direction_bonus,
        'action_penalty': transformed_action_penalty
    }

    return total_reward, reward_components
