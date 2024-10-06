@torch.jit.script
def compute_reward(root_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the improved reward for the humanoid task of running as fast as possible.

    Inputs:
    - root_states: Tensor containing the state of the humanoid (position and velocity).

    Returns:
    - reward: Total reward based on horizontal speed and balance.
    - reward_components: Dictionary of individual reward components.
    """
    
    # Constants
    speed_temp = 0.05   # Temperature for speed reward normalization
    balance_temp = 0.1   # Temperature for stabilization reward normalization

    # Extract torso velocity and angular velocity for stability
    torso_velocity = root_states[:, 7:10]  # [vx, vy, vz]
    torso_rotation = root_states[:, 3:7]    # [qx, qy, qz, qw]
    
    # Calculate forward speed (along the x-axis)
    forward_speed = torso_velocity[:, 0]
    speed_reward = torch.exp(forward_speed / speed_temp)

    # Calculate balance based on the magnitude of angular velocity (to keep humanoid stable)
    ang_velocity = torch.norm(torso_velocity[:, 10:13], p=2, dim=1)
    balance_reward = torch.exp(1.0 / (1.0 + ang_velocity))  # Reward for less angular velocity

    # Total reward is a combination of speed reward and balance reward
    total_reward = speed_reward + balance_reward

    # Creating the reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'balance_reward': balance_reward
    }

    return total_reward, reward_components
