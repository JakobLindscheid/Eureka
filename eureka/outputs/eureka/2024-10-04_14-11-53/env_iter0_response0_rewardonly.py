@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Calculate the forward movement (x direction)
    forward_speed = velocity[:, 0]  # Assuming forward movement is along the x-axis
    
    # Calculate distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)

    # Define temperature parameters for normalization
    forward_temp: float = 0.1
    distance_temp: float = 0.1

    # Reward components
    forward_reward = torch.exp(forward_temp * forward_speed)  # Reward for moving forward
    distance_reward = torch.exp(distance_temp * (-distance_to_target))  # Reward for getting closer to the target

    # Total reward
    total_reward = forward_reward + distance_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'distance_reward': distance_reward
    }

    return total_reward, reward_components
