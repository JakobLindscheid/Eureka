@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso position and its velocity
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Calculate forward speed
    forward_speed = velocity[:, 0]
    
    # Calculate the distance from the target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Euclidean distance

    # Reward weights and temperature parameters
    speed_threshold: float = 1.0 
    distance_reward_scale: float = 5.0
    speed_temp: float = 0.5  # Temperature for forward speed
    distance_temp: float = 0.2  # Temperature for distance

    # Reward for forward speed using a softplus to ensure positivity
    forward_reward = torch.log1p(forward_speed / speed_threshold)  # Smoother scaling for speed
    
    # New distance reward - incentivizes getting closer to the target
    distance_reward = distance_reward_scale * torch.exp(-distance_temp * distance_to_target)

    # Total reward as a combination
    total_reward = forward_reward + distance_reward

    # Individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'distance_reward': distance_reward
    }

    return total_reward, reward_components
