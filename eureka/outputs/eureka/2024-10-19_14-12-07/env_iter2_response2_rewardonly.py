@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for scaling rewards
    speed_temp: float = 1.0
    action_temp: float = 0.1
    
    # Extract velocity from root states (assuming root_states contains [position, rotation, velocity, angular_velocity])
    velocity = root_states[:, 7:10]  # Extract velocity part

    # Compute the speed (magnitude of the velocity vector)
    speed = torch.norm(velocity, p=2, dim=1)

    # Reward based on speed
    speed_reward = torch.exp(speed / speed_temp)
    
    # Penalty for excessive control actions (to discourage unnecessary movements)
    action_penalty = -torch.norm(actions, p=2, dim=1)  # Negative reward for large actions
    action_reward = torch.exp(action_penalty / action_temp)
    
    # Total reward combines speed reward and action reward
    total_reward = speed_reward + action_reward

    # Individual reward components
    reward_components = {
        'speed_reward': speed_reward,
        'action_reward': action_reward
    }

    return total_reward, reward_components
