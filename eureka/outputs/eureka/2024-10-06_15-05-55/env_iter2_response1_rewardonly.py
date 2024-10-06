@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the humanoid task of running as fast as possible with improved scaling.

    Inputs:
    - root_states: Tensor containing the state of the humanoid (position and velocity).
    - targets: Tensor containing the target position for the humanoid.
    - dt: Time step for the environment.

    Returns:
    - reward: Total reward based on the humanoid's speed and progress towards the target.
    - reward_components: Dictionary of individual reward components.
    """
    
    # Constants
    speed_temp = 0.5  # Temperature for speed normalization
    progress_temp = 10.0  # Temperature for forward progress normalization
    task_score_temp = 1.0  # Temperature for task score normalization

    # Extracting the torso velocity and position
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_position = root_states[:, 0:3]  # [x, y, z]
    
    # Calculate forward speed (along the x-axis for running)
    forward_speed = torso_velocity[:, 0]  # Only take the x component of velocity
    speed_reward = (forward_speed / (torch.norm(forward_speed) + 1e-5)) ** 2  # Normalize and square for better scaling

    # Calculate forward progress toward the target
    direction_to_target = targets[:, 0] - torso_position[:, 0]
    progress_reward = torch.norm(direction_to_target) / (dt + 1e-5)  # Normalized reward

    # Compute task score (currently using a placeholder function, modify as needed)
    task_score_reward = 1.0 / (torch.abs(forward_speed) + 1e-5)
    task_score_reward = torch.log(task_score_reward + 1.0) / task_score_temp  # Log scale application

    # Total reward is a combination of speed reward, forward progress reward, and task score
    total_reward = speed_reward + progress_reward + task_score_reward

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'forward_progress_reward': progress_reward,
        'task_score_reward': task_score_reward
    }

    return total_reward, reward_components
