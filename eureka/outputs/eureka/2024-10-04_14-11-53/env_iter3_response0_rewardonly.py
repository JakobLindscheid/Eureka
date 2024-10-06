@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float, distance_threshold: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Calculate the forward movement (in the x direction)
    forward_speed = velocity[:, 0]

    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance

    # Define constants for normalizing reward components
    speed_threshold = 1.0  # Threshold for meaningful forward speed
    successful_proximity = distance_threshold  # Reward if within this distance

    # Define temperature parameters for normalization
    forward_temp: float = 0.2  # Temperature for forward reward transformation
    proximity_temp: float = 0.1  # Temperature for proximity reward transformation
    
    # Forward Reward: Normalize forward speed with an exponential function
    forward_reward = torch.clamp(torch.exp(forward_speed / forward_temp) - 1, min=0)
    
    # Proximity Reward: Reward for reducing distance to the target
    proximity_reward = torch.where(
        distance_to_target < successful_proximity, 
        (successful_proximity - distance_to_target) / successful_proximity, 
        torch.tensor(0.0, device=root_states.device)
    )

    # Task Score: Combine both forward reward and proximity to target
    task_score = forward_reward * 0.5 + proximity_reward * 0.5

    # Total reward - combining all components
    total_reward = forward_reward + proximity_reward + task_score

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
