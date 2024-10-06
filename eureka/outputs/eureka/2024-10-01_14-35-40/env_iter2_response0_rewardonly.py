@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters for temperature scaling
    running_temp = 0.1
    action_temp = 0.5
    task_temp = 0.2

    # Extract forward velocity and compute running reward
    velocity = root_states[:, 7:10]  # x, y, z velocities
    forward_velocity = velocity[:, 0]  
    running_reward = torch.exp(forward_temp * forward_velocity)  # Exponential scaling
    
    # Penalty for excessive actions (stronger penalty)
    action_penalty = -torch.exp(action_temp * (torch.norm(actions, p=2, dim=-1) ** 2))
    
    # Calculate distance to the target for task score
    torso_position = root_states[:, 0:3]  # Torso position x, y, z
    distance_to_target = torch.norm(targets - torso_position, p=2, dim=1)
    task_score = torch.exp(-task_temp * distance_to_target)  # Closer is better, increasing reward

    # Define the total reward
    total_reward = running_reward + action_penalty + task_score

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty,
        "task_score": task_score
    }
    
    return total_reward, reward_components
