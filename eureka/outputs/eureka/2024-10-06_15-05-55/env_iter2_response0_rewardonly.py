@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float, episode_length: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the humanoid task of running as fast as possible with improved scaling.

    Inputs:
    - root_states: Tensor containing the state of the humanoid (position and velocity).
    - targets: Tensor containing the target position for the humanoid.
    - dt: Time step for the environment.
    - episode_length: Length of the current episode.

    Returns:
    - reward: Total reward based on speed, forward progress, and task score.
    - reward_components: Dictionary of individual reward components.
    """
    
    # Constants
    speed_temp = 2.0  # Adjustable temperature for speed normalization
    progress_temp = 2.0  # Temperature for progress normalization
    task_temp = 0.1  # Increased temperature for task score normalization
    
    # Extracting the torso velocity and position
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_position = root_states[:, 0:3]  # [x, y, z]
    
    # Calculate forward speed (along the x-axis for running)
    forward_speed = torso_velocity[:, 0]  # Only take the x component of velocity
    speed_reward = torch.exp(forward_speed / speed_temp)  # Using exponential for positive scaling

    # Calculate total distance moved toward the target
    distance_to_target = targets[:, 0] - torso_position[:, 0]  # x-coordinate
    previous_distance = distance_to_target  # To calculate distance covered; in a real scenario, this would require tracking
    progress_reward = torch.exp((previous_distance) / progress_temp) - 1.0  # Use exponential to normalize reward
    
    # Calculate task score as a vector to provide more variation
    task_score = torch.norm(torso_velocity, p=2, dim=-1) / (episode_length + 1e-5)  # Normalize with episode length
    
    # Total reward is a combination of normalized speed, forward progress reward, and task score
    total_reward = speed_reward + progress_reward + (task_score / task_temp)
    
    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'forward_progress_reward': progress_reward,
        'task_score': task_score
    }

    return total_reward, reward_components
