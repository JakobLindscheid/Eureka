@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, previous_positions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the humanoid task of running as fast as possible with improved scaling.

    Inputs:
    - root_states: Tensor containing the state of the humanoid (position and velocity).
    - targets: Tensor containing the target position for the humanoid.
    - previous_positions: Tensor containing the previous position of the humanoid.
    - dt: Time step for the environment.

    Returns:
    - reward: Total reward based on the humanoid's speed, progress towards the target, and other factors.
    - reward_components: Dictionary of individual reward components.
    """
    
    # Constants
    speed_temp = 2.0  # Temperature for speed normalization
    progress_temp = 5.0  # Temperature for progress reward normalization
    task_score_temp = 3.0  # Temperature for task score normalization
    episode_length_reward_factor = 0.01  # Factor for episode length contribution

    # Extracting torso velocity and position
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_position = root_states[:, 0:3]  # [x, y, z]
    
    # Calculate forward speed (along the x-axis for running) and normalize
    forward_speed = torch.clamp(torso_velocity[:, 0], min=0)  # Only take the x component, prevent negatives
    speed_reward = torch.log(forward_speed + 1e-5) / speed_temp  # Log scaling
    
    # Calculate distance moved toward the target (forward progress)
    forward_progress = torso_position[:, 0] - previous_positions[:, 0]  # Change in x position
    forward_progress_reward = forward_progress / (torch.norm(forward_progress) + 1e-5)  # Normalize to prevent division by zero

    # Calculate task score based on progress toward the target
    distance_to_target = torch.norm(targets[:, :2] - torso_position[:, :2], p=2)  # 2D distance on x, y plane to target
    task_score = 5.0 / (distance_to_target + 1e-5)  # Estimate score close to task completion

    # Calculate episode length reward
    episode_length_reward = episode_length_reward_factor * (dt)  # Scale based on episode length in time

    # Total reward is a combination of the rewards
    total_reward = speed_reward + forward_progress_reward + task_score + episode_length_reward

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'forward_progress_reward': forward_progress_reward,
        'task_score': task_score,
        'episode_length_reward': episode_length_reward
    }

    return total_reward, reward_components
