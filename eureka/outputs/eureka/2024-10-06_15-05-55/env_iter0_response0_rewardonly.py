@torch.jit.script
def compute_reward(velocity: torch.Tensor, target_velocity: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward components
    speed_reward: float = 1.0  # Base weight for speed reward
    action_penalty: float = 0.01  # Weight for action penalty

    # Calculate the speed towards the target
    speed = torch.norm(velocity, p=2, dim=-1)
    target_speed = torch.norm(target_velocity, p=2, dim=-1)
    
    # Reward for running fast
    running_reward = speed - target_speed  # Encourage agent to exceed the target speed
    running_reward = torch.exp(running_reward)  # Transform to create a larger range of rewards

    # Penalty for high action values
    action_penalty_value = torch.sum(torch.abs(action)) * action_penalty
    action_penalty_value = torch.exp(-action_penalty_value)  # Transform to encourage minimal action

    # Total reward
    total_reward = running_reward - action_penalty_value

    # Creating a rewards dictionary for breakdown
    rewards_dict = {'running_reward': running_reward, 'action_penalty': action_penalty_value}

    return total_reward, rewards_dict
