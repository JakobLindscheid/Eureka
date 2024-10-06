@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.5
    alignment_temp = 0.5
    proximity_temp = 0.5  # New temperature for proximity reward

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component
    forward_velocity = velocity[:, 0]

    # Reward component for moving forward quickly (scaled up)
    speed_reward = torch.clamp(forward_velocity / 5.0, min=0.0)  # Increased scale for speed reward

    # Calculate distance to target
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)

    # Compute proximity reward (incentivizing closeness to the target)
    proximity_reward = torch.exp(-distance_to_target / proximity_temp)  # Exponential decay as a reward

    # Normalize direction to target for alignment reward
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / 5.0, min=0.0)  # Increased scale

    # Total reward includes all components
    total_reward = torch.exp(speed_reward / speed_temp) + torch.exp(alignment_reward / alignment_temp) + proximity_reward

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'proximity_reward': proximity_reward,
    }
    
    return total_reward, reward_components
