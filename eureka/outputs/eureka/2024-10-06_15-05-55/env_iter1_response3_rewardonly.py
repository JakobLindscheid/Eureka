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
    speed_temp = 5.0  # Increased temperature for speed normalization
    progress_temp = 5.0  # Increased temperature for backward progress normalization

    # Extracting the torso velocity and position
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_position = root_states[:, 0:3]  # [x, y, z]
    
    # Calculate forward speed (along the x-axis for running)
    forward_speed = torso_velocity[:, 0]  # Only take the x component of velocity
    speed_reward = forward_speed / speed_temp  # Linear scaling of speed

    # Calculate progress towards the target on the x-axis
    progress_toward_target = targets[:, 0] - torso_position[:, 0]
    forward_progress_reward = progress_toward_target / (torch.norm(progress_toward_target) + 1e-5)  # Normalize progress

    # Total reward is a combination of normalized speed reward and forward progress reward
    total_reward = speed_reward + forward_progress_reward

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'forward_progress_reward': forward_progress_reward
    }

    return total_reward, reward_components
