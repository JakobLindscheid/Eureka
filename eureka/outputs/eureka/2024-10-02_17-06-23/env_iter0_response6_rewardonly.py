@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define a temperature variable for reward scaling
    temperature_position = 0.1
    temperature_velocity = 0.1
    
    # Extract torso position and velocity from root states
    torso_position = root_states[:, 0:3]  # Shape: (N, 3)
    velocity = root_states[:, 7:10]        # Shape: (N, 3)

    # Calculate distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)  # Shape: (N,)
    
    # Reward components
    reward_forward = -distance_to_target  # Negative distance as we want to minimize it
    reward_velocity = torch.norm(velocity[:, :2], p=2, dim=-1)  # Forward velocity (ignoring vertical)

    # Normalize rewards using temperature
    total_reward = torch.exp(reward_forward / temperature_position) + torch.exp(reward_velocity / temperature_velocity)
    
    # Create reward components dictionary
    reward_components = {
        'reward_forward': reward_forward,
        'reward_velocity': reward_velocity
    }

    return total_reward, reward_components
