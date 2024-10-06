@torch.jit.script
def compute_reward(dof_vel: torch.Tensor, target_velocity: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Set temperature for the reward transformation
    velocity_temp = 0.1  # This can be adjusted based on desired sensitivity

    # Calculate the speed of the humanoid
    speed = torch.norm(dof_vel[:, 0:2], p=2, dim=-1)  # Only consider x and y components for running speed

    # Reward proportional to the speed towards the target velocity
    speed_to_target_reward = speed - target_velocity
    reward = speed_to_target_reward

    # Transform the reward for normalization
    transformed_reward = torch.exp(velocity_temp * reward) - 1

    # Create a dictionary for individual reward components
    reward_components = {
        'speed_to_target_reward': speed_to_target_reward,
        'transformed_reward': transformed_reward
    }

    return transformed_reward, reward_components
