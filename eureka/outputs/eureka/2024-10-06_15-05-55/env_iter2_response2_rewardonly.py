@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the humanoid task of running as fast as possible with improved components.

    Inputs:
    - root_states: Tensor containing the state of the humanoid (position and velocity).
    - targets: Tensor containing the target position for the humanoid.
    - dt: Time step for the environment.

    Returns:
    - reward: Total reward based on the humanoid's speed and forward progress.
    - reward_components: Dictionary of individual reward components including normalized speed and progress rewards.
    """
    
    # Constants
    speed_temp = 1.0  # Temperature for speed normalization
    progress_temp = 1.0  # Temperature for progress normalization
    task_temp = 2.0  # Temperature for task score scaling

    # Extracting the torso velocity and position
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_position = root_states[:, 0:3]  # [x, y, z]

    # Calculate forward speed (along the x-axis for running)
    forward_speed = torso_velocity[:, 0]  # Only take the x component of velocity
    normalized_speed = torch.clamp(forward_speed / (torch.norm(torso_velocity, dim=-1) + 1e-5), min=-1.0, max=1.0)
    speed_reward = torch.exp(normalized_speed / speed_temp)

    # Calculate distance covered towards the target on the x-axis
    progress_toward_target = targets[:, 0] - torso_position[:, 0]
    normalized_progress = progress_toward_target / (torch.norm(progress_toward_target) + 1e-5)  # Normalize progress
    progress_reward = torch.exp(normalized_progress / progress_temp)

    # Transforming the task score as a function of success
    task_score = speed_reward + progress_reward
    
    # Total reward is a combination of speed, forward progress, and transformed task score
    total_reward = speed_reward + progress_reward + task_score

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'progress_reward': progress_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
