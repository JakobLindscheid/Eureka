@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 1.0
    alignment_temp = 1.0
    distance_temp = 0.5

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Forward velocity component (scaling to make it more sensitive)
    forward_velocity = velocity[:, 0]
    speed_reward = torch.pow(torch.clamp(forward_velocity, min=0.0), 2) / 10.0  # Greater sensitivity to speed increases the incentive

    # Calculate the direction towards the target
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)

    # New reward component for reducing distance to target, optimized for negative distances
    distance_reward = torch.exp(-distance_to_target / distance_temp)

    # Alignment reward (focus more on achieving better directional alignment)
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)  # Prevent division by zero
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / (torch.norm(velocity, p=2, dim=-1) + 1e-6), min=0.0)

    # Total reward is the combination of all necessary components
    total_reward = speed_reward + alignment_reward + distance_reward
    
    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'distance_reward': distance_reward,
    }
    
    return total_reward, reward_components
