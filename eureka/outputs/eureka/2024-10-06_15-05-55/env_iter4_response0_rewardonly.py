@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the humanoid task focused on running as fast as possible with improved components.

    Inputs:
    - root_states: Tensor containing the state of the humanoid (position and velocity).
    - targets: Tensor containing the target position for the humanoid.
    - dt: Time step for the environment.

    Returns:
    - reward: Total reward based on the humanoid's speed, progress towards the target, and scaled task score.
    - reward_components: Dictionary of individual reward components.
    """

    # Define temperature parameters for normalization
    speed_temp = 20.0  # Temperature for speed normalization
    forward_progress_temp = 5.0  # Temperature for forward progress normalization
    task_score_temp = 10.0  # Temperature for task score normalization

    # Extracting torso state
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_position = root_states[:, 0:3]  # [x, y, z]

    # Calculate forward speed (only in x-axis) and ensure positive reward
    forward_speed = torso_velocity[:, 0]
    normalized_speed_reward = torch.clamp(forward_speed, min=0.0)  # Ensure non-negative values

    # Calculate forward progress toward the target in x-axis
    distance_to_target = targets[:, 0] - torso_position[:, 0]  # x-component only
    forward_progress_reward = torch.clamp(distance_to_target, min=0.0)  # Capture actual distance covered

    # Normalize forward progress
    normalized_forward_progress = forward_progress_reward / (torch.norm(distance_to_target) + 1e-5)

    # Normalize task score 
    overall_task_score = normalized_speed_reward + normalized_forward_progress
    normalized_task_score = torch.clamp(overall_task_score / (task_score_temp + 1e-5), min=0.0, max=1.0)

    # Summing all rewards with respective scaling
    total_reward = (
        torch.exp(normalized_speed_reward / speed_temp) + 
        torch.exp(normalized_forward_progress / forward_progress_temp) + 
        normalized_task_score
    )

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'forward_progress_reward': normalized_forward_progress,
        'task_score': normalized_task_score
    }

    return total_reward, reward_components
