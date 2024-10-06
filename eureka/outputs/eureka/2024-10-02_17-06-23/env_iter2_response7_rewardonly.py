@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 1.0
    alignment_temp = 0.5
    target_reach_temp = 0.5
    
    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component
    forward_velocity = velocity[:, 0]

    # Reward component for forward speed (without clamping)
    speed_reward = forward_velocity / 10.0  # Adjust scale as needed
    
    # Calculate the direction to target
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)
    
    # Normalize direction
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)

    # Alignment reward based on how closely the agent's velocity is aligned with the direction to the target
    alignment_reward = torch.sum(velocity * direction_to_target_normalized, dim=-1) / (torch.norm(velocity, p=2, dim=-1) + 1e-6)

    # New target reach reward based on the inverse of the distance to target
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)  # Reward for getting closer to the target
    
    # Total reward
    total_reward = speed_reward + alignment_reward + target_reach_reward
    
    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'target_reach_reward': target_reach_reward,
    }
    
    return total_reward, reward_components
