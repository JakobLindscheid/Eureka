@torch.jit.script
def compute_reward(velocity: torch.Tensor, target_velocity: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature variables for normalization
    temp_velocity = 1.0
    temp_time = 1.0

    # Calculate the speed (magnitude) of the humanoid's velocity
    speed = torch.norm(velocity, p=2, dim=-1)

    # Calculate the reward based on speed aiming to maximize it over time
    reward_speed = speed / dt
    reward_speed_transformed = torch.exp(reward_speed / temp_velocity)

    # Reward for staying close to the target speed (assumed to be max speed, adjust as needed)
    target_speed = torch.norm(target_velocity, p=2, dim=-1)
    reward_target_speed = -torch.abs(target_speed - speed)
    reward_target_speed_transformed = torch.exp(reward_target_speed / temp_time)

    # Total reward is a combination of speed reward and target speed reward
    total_reward = reward_speed_transformed + reward_target_speed_transformed

    return total_reward, {
        'speed_reward': reward_speed_transformed,
        'target_speed_reward': reward_target_speed_transformed
    }
