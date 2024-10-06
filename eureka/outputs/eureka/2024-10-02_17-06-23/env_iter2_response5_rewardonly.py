@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 0.3
    alignment_temp = 0.4
    distance_temp = 0.2
    idle_penalty_temp = 0.5
    
    # Extract necessary components
    torso_position = root_states[:, 0:3]  # Shape: (num_agents, 3)
    velocity = root_states[:, 7:10]  # Shape: (num_agents, 3)
    
    # Forward velocity component
    forward_velocity = velocity[:, 0]  # X-axis velocity

    # Reward for moving forward quickly (normalized)
    speed_reward = torch.exp(forward_velocity / speed_temp) - 1  # Exponential growth reduces at small speeds
    
    # Direction towards the target
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)
    direction_to_target_normalized = direction_to_target / (distance_to_target.unsqueeze(-1) + 1e-6)  # Normalize

    # Angle to target (replacing alignment reward)
    angle_to_target = torch.acos(torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / 
                                               (torch.norm(velocity, p=2, dim=-1) + 1e-6), -1.0, 1.0))
    angle_reward = torch.exp(-angle_to_target / alignment_temp)

    # Reward for decreasing distance to target
    target_reach_reward = -distance_to_target / distance_temp  # Encourage reducing the distance

    # Reward penalizing minimal movement (idling)
    idle_penalty = -torch.norm(dof_vel, p=2, dim=1) / idle_penalty_temp  # Penalize lack of movement

    # Total reward is the sum of all components
    total_reward = speed_reward + angle_reward + target_reach_reward + idle_penalty
    
    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': angle_reward,
        'target_reach_reward': target_reach_reward,
        'idle_penalty': idle_penalty,
    }
    
    return total_reward, reward_components
