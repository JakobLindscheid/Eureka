@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, vel_loc: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso velocity
    torso_velocity = root_states[:, 7:10]  # Assuming velocity is at this index
    speed = torch.norm(torso_velocity, p=2, dim=-1)
    
    # Reward for speed, encourage running fast
    speed_reward = speed
    
    # Penalize for excessive actions (to encourage efficient movements)
    action_penalty = -torch.norm(actions, p=2, dim=-1)

    # Normalize rewards using temperature
    speed_temp = 0.1
    action_temp = 0.05

    normalized_speed_reward = torch.exp(speed_temp * speed_reward)
    normalized_action_penalty = torch.exp(action_temp * action_penalty)

    # Total reward is speed reward minus the action penalty
    total_reward = normalized_speed_reward + normalized_action_penalty

    # Create a dictionary for individual components
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'action_penalty': normalized_action_penalty
    }

    return total_reward, reward_components
