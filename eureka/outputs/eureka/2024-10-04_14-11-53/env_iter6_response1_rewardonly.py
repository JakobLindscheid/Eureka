@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Forward movement reward (x direction)
    forward_speed = velocity[:, 0]
    forward_temp = 1.0  # Adjusting temperature for forward speed scaling
    forward_reward = torch.exp(forward_speed / forward_temp) - 1  # Exponential scaling

    # New proximity reward: positive reward as the agent gets closer to the target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance
    max_distance = 10.0  # Maximum distance for scaling purposes
    proximity_temp = 2.0  # Temperature for proximity reward scaling
    proximity_reward = torch.exp((max_distance - distance_to_target) / proximity_temp) - 1

    # Task reward: provide reward based on success in the environment
    task_success = torch.where(distance_to_target < 1.0, torch.tensor(1.0, device=root_states.device), torch.tensor(0.0, device=root_states.device))

    # Total reward
    total_reward = forward_reward + proximity_reward + task_success

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'proximity_reward': proximity_reward,
        'task_success': task_success
    }

    return total_reward, reward_components
