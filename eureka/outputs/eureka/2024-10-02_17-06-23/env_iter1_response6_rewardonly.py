@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 0.5
    alignment_temp = 0.5
    task_multiplier = 5.0  # Reward multiplier for task achievement

    # Extract torso position and velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate the forward velocity component
    forward_velocity = velocity[:, 0] 

    # Reward for moving forward quickly, normalize and introduce a scaling factor
    speed_reward = torch.exp(forward_velocity / speed_temp)

    # Calculate the direction to the target
    direction_to_target = targets - torso_position
    direction_to_target[:, 2] = 0.0 
    direction_to_target_normalized = direction_to_target / (torch.norm(direction_to_target, p=2, dim=-1, keepdim=True) + 1e-6)

    # Compute alignment reward, normalize, and introduce a scaling factor
    alignment_reward = torch.exp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / alignment_temp)

    # Introduce a task score based on distance covered towards the target
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1)
    task_score = -distance_to_target + task_multiplier * (torch.sigmoid(forward_velocity))

    # Total reward combines speed, alignment, and task score
    total_reward = (0.5 * speed_reward + 0.5 * alignment_reward) + task_score

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'task_score': task_score,
    }
    
    return total_reward, reward_components
