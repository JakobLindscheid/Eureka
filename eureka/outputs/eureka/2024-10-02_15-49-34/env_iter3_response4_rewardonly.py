@torch.jit.script
def compute_reward(root_states: torch.Tensor, action: torch.Tensor, dt: float, 
                   target_velocity: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extracting the position and velocity
    torso_velocity = root_states[:, 7:10]  # assuming the velocity is in the range [7:10]
    forward_velocity = torso_velocity[:, 0]  # only the x-component for forward velocity

    # Reward components
    velocity_reward = forward_velocity / target_velocity  # Reward based on the forward velocity
    action_reward = -torch.norm(action, p=2, dim=-1)  # Penalize large actions
    total_reward = velocity_reward + action_reward

    # Normalize the reward components
    temperature_velocity = 0.1  # temperature for velocity reward normalization
    temperature_action = 0.1  # temperature for action penalty normalization
    
    normalized_velocity_reward = torch.exp(temperature_velocity * velocity_reward)
    normalized_action_reward = torch.exp(temperature_action * action_reward)

    # Total normalized reward
    total_normalized_reward = normalized_velocity_reward + normalized_action_reward

    # Dictionary of individual reward components
    reward_components = {
        'velocity_reward': normalized_velocity_reward,
        'action_reward': normalized_action_reward
    }

    return total_normalized_reward, reward_components
