@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the humanoid task of running as fast as possible with improved components and scaling.
    
    Inputs:
    - root_states: Tensor containing the state of the humanoid (position and velocity).
    - targets: Tensor containing the target position for the humanoid.
    - dt: Time step for the environment.
    
    Returns:
    - reward: Total reward based on the humanoid's speed, progress towards the target, and task score.
    - reward_components: Dictionary of individual reward components.
    """
    
    # Temperature parameters
    speed_temp = 0.5  # Temperature for speed normalization
    progress_temp = 1.0  # Temperature for forward progress normalization
    task_score_temp = 1000.0  # Temperature for task score normalization
    
    # Extracting the torso velocity and position
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_position = root_states[:, 0:3]  # [x, y, z]

    # Calculate forward speed (only in x-axis)
    forward_speed = torso_velocity[:, 0]
    normalized_speed_reward = torch.exp(forward_speed / speed_temp) - 1  # Exponential to promote positive learning

    # Calculate forward progress based on distance moved towards the target
    current_target_position = targets.clone()  # Copy targets to avoid modification during learning
    target_progress = current_target_position[:, 0] - torso_position[:, 0]  
    previous_progress = torch.clamp(torch.norm(target_progress, p=2), min=1e-5)  # Avoid division by zero
    forward_progress_reward = (torch.norm(torso_velocity, p=2) / previous_progress) * dt  # Proportional to velocity and time step

    # Updated task score - keeping it in a manageable range
    task_score = normalized_speed_reward + forward_progress_reward
    normalized_task_score = torch.clamp(torch.exp(task_score / task_score_temp) - 1, min=0.0, max=100.0)  # Clamped for stability

    # Total reward
    total_reward = normalized_speed_reward + forward_progress_reward + normalized_task_score

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'forward_progress_reward': forward_progress_reward,
        'task_score': normalized_task_score
    }

    return total_reward, reward_components
