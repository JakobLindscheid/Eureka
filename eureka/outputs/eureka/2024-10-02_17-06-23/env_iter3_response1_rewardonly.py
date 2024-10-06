@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming reward components
    speed_temp = 0.2
    target_reach_temp = 0.5
    movement_penalty_temp = 0.1
    task_complete_temp = 0.3

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Compute the forward velocity (only x-axis)
    forward_velocity = velocity[:, 0]

    # Adjust the speed reward for normalization
    speed_reward = torch.clamp(forward_velocity ** 2 / 25.0, min=0.0, max=1.0)  # Squared and normalized
    speed_reward_transformed = torch.exp(speed_reward / speed_temp)

    # Calculate the distance to the target
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)

    # New reward for approaching the target
    target_reach_reward = torch.clamp(1.0 - (distance_to_target / torch.max(distance_to_target)), min=0.0, max=1.0)
    target_reach_reward_transformed = torch.exp(target_reach_reward / target_reach_temp)

    # Movement penalty for staying still
    no_movement_penalty = -torch.exp(-torch.norm(velocity, p=2, dim=-1) / movement_penalty_temp)

    # Introduce a task completion reward, e.g., reaching a significant distance or goal
    task_completion_reward = torch.clamp((25.0 - distance_to_target) / 25.0, min=0.0, max=1.0)
    task_completion_reward_transformed = torch.exp(task_completion_reward / task_complete_temp)

    # Total reward is a combination of all components
    total_reward = speed_reward_transformed + target_reach_reward_transformed + no_movement_penalty + task_completion_reward_transformed

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward_transformed,
        'target_reach_reward': target_reach_reward_transformed,
        'movement_penalty': no_movement_penalty,
        'task_completion_reward': task_completion_reward_transformed
    }
    
    return total_reward, reward_components
