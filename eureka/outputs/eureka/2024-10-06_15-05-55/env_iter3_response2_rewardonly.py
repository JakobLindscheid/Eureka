@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the humanoid task of running as fast as possible with improved components and scaling.

    Inputs:
    - root_states: Tensor containing the state of the humanoid (position and velocity).
    - targets: Tensor containing the target position for the humanoid.
    - dt: Time step for the environment.

    Returns:
    - reward: Total reward based on the humanoid's speed and task score.
    - reward_components: Dictionary of individual reward components.
    """
    
    # Temperature parameters
    speed_temp = 1.0  # Temperature for speed normalization
    task_score_temp = 1e10  # Temperature for task score normalization

    # Extracting the torso velocity and position
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_position = root_states[:, 0:3]  # [x, y, z]

    # Calculate forward speed (only in x-axis)
    forward_speed = torso_velocity[:, 0]  

    # Clamping the speed reward to ensure no negative effects
    normalized_speed_reward = torch.clamp(forward_speed, min=0.0)  # Ensure non-negativity

    # Simplified task score calculation (normalized)
    distance_to_target = targets[:, 0] - torso_position[:, 0]  # x-component only
    task_score = normalized_speed_reward + torch.norm(distance_to_target, p=2)  # Adding distance to target
    normalized_task_score = torch.exp(task_score / task_score_temp) - 1  # Transform to avoid negatives

    # Total reward
    total_reward = normalized_speed_reward + normalized_task_score

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'task_score': normalized_task_score
    }

    return total_reward, reward_components
