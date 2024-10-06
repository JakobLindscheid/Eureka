@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 1.0
    alignment_temp = 0.5
    target_reach_temp = 0.5

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Forward velocity component (as a fraction of a threshold)
    forward_velocity = velocity[:, 0]
    speed_reward = torch.clamp(forward_velocity / (1.0 + torch.abs(forward_velocity)), min=0.0)  # Normalize speed reward dynamically

    # Calculate the direction towards the target
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0  # Ignore z-axis
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)

    # Reward component for reaching the target (incremental based on proximity)
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)

    # Alignment with target direction (angle measure)
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / (torch.norm(velocity, p=2, dim=-1) + 1e-6), min=0.0)

    # Total reward is the combination of all components
    total_reward = torch.exp(speed_temp * speed_reward + alignment_temp * alignment_reward + target_reach_temp * target_reach_reward)

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'target_reach_reward': target_reach_reward,
    }
    
    return total_reward, reward_components
