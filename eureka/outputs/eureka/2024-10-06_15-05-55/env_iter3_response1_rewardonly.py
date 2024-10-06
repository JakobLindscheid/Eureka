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
    speed_temp = 1.0  # Lower temperature for more stable speed normalization
    task_score_temp = 5.0  # Adjusted temperature for task score normalization

    # Extracting the torso velocity and position
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]

    # Calculate forward speed (only in x-axis)
    forward_speed = torso_velocity[:, 0]  
    normalized_speed_reward = torch.clip(forward_speed, min=0)  # Encouraging positive speed only

    # Calculate task score
    # Assuming task score starts from something derived from speed and observation
    task_score = normalized_speed_reward.mean() + 0.01 

    # Normalizing task score to prevent large numbers
    normalized_task_score = torch.exp(task_score / task_score_temp) - 1  

    # Total reward summation
    total_reward = normalized_speed_reward + normalized_task_score

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': normalized_speed_reward,
        'task_score': normalized_task_score
    }

    return total_reward, reward_components
