@torch.jit.script
def compute_reward(torso_velocity: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for running speed (magnitude of torso velocity)
    speed_reward_temp = 0.1  # Temperature for speed reward
    speed_reward = torch.norm(torso_velocity, p=2, dim=-1)  # Magnitude of torso velocity
    speed_reward_transformed = torch.exp(speed_reward / speed_reward_temp)  # Transformed speed reward

    # Penalize excessive joint velocities (to prevent wild movements)
    velocity_penalty_temp = 0.1  # Temperature for velocity penalty
    velocity_penalty = torch.sum(dof_vel ** 2, dim=-1)  # Sum of squared joint velocities
    velocity_penalty_transformed = torch.exp(-velocity_penalty / velocity_penalty_temp)  # Transformed penalty

    # Total reward is the difference of speed reward and velocity penalty
    total_reward = speed_reward_transformed + velocity_penalty_transformed

    # Create a dictionary of individual reward components
    reward_components = {
        "speed_reward": speed_reward_transformed,
        "velocity_penalty": velocity_penalty_transformed
    }

    return total_reward, reward_components
