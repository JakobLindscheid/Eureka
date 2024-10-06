@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]
    forward_temp = 0.5  # temperature parameter for forward reward scaling
    forward_reward = torch.clamp(torch.exp(forward_speed / forward_temp) - 1, min=0.0)  # Ensure no negative rewards

    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance

    # New proximity reward: provide reward based on distance from target with an exponential function
    proximity_temp = 0.5  # Temperature parameter for proximity reward scaling
    max_proximity_reward = 5.0  # Maximum reward for being at the target
    proximity_reward = max_proximity_reward * torch.exp(-distance_to_target / proximity_temp)

    # Task score providing incentive for the number of successful steps per episode
    task_temp = 1.0  # temperature for task score
    successful_threshold = 1.0
    task_score = torch.where(forward_speed > successful_threshold, 
                             1.0, 
                             0.0)  # Reward for achieving a certain speed
    
    # Total reward
    total_reward = forward_reward + proximity_reward + task_score

    # Dictionary of individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
