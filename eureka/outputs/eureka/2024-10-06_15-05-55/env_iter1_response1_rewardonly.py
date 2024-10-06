@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the humanoid task of running efficiently.

    Inputs:
    - root_states: Tensor containing the state of the humanoid (position and velocity).
    - targets: Tensor containing the target position for the humanoid.
    - dt: Time step for the environment.

    Returns:
    - total_reward: Combined reward based on speed, forward progress, and directional consistency.
    - reward_components: Dictionary of individual reward components.
    """
    
    # Constants
    speed_temp = 0.5  # Adjusted temperature for speed normalization
    consistency_temp = 0.2  # Temperature for direction consistency reward
    
    # Extracting the torso velocity and position
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_position = root_states[:, 0:3]  # [x, y, z]
    target_position = targets[:, 0]  # x component of the target
    
    # Calculate forward speed (along the x-axis for running)
    forward_speed = torso_velocity[:, 0]  # Only take the x component of velocity
    speed_reward = torch.exp(forward_speed / speed_temp)  # Exponentially scale speed

    # Calculate target directional angle and consistency
    direction_to_target = torch.atan2(target_position - torso_position[:, 0], torso_position[:, 1])
    heading_angle = torch.atan2(torso_velocity[:, 1], torso_velocity[:, 0])
    direction_consistency_reward = torch.exp(-torch.abs(heading_angle - direction_to_target) / consistency_temp)

    # Total reward is a combination of speed reward, and direction consistency reward
    total_reward = speed_reward + direction_consistency_reward

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'direction_consistency_reward': direction_consistency_reward
    }

    return total_reward, reward_components
