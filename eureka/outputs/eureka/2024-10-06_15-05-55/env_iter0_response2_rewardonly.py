@torch.jit.script
def compute_reward(current_velocity: torch.Tensor, previous_velocity: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature variable for reward transformation
    velocity_temp: float = 0.1

    # Calculate the speed (magnitude of velocity vector)
    speed = torch.norm(current_velocity, p=2, dim=-1)

    # Reward for increasing speed compared to previous velocity
    speed_increase = speed - torch.norm(previous_velocity, p=2, dim=-1)

    # Total reward: reward for speed increase normalized by temperature
    total_reward = torch.exp(speed_increase / velocity_temp)

    # Create a dictionary to hold individual reward components
    reward_components = {
        'speed_increase': speed_increase,
        'total_speed_reward': total_reward,
    }

    return total_reward, reward_components
