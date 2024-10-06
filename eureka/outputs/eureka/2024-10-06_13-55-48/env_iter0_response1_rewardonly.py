@torch.jit.script
def compute_reward(velocity: torch.Tensor, prev_potentials: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature for reward scaling
    temperature_speed = 0.1 
    temperature_potential = 0.1

    # Reward component based on the speed (magnitude of velocity)
    speed = torch.norm(velocity, p=2, dim=-1)
    speed_reward = torch.exp(temperature_speed * speed)

    # Reward component based on potential change (encouraging continuous running)
    potential_change = prev_potentials - speed
    potential_reward = torch.exp(temperature_potential * potential_change)

    # Total reward combining speed and potential rewards
    total_reward = speed_reward + potential_reward

    # Create a dictionary of individual reward components
    reward_components = {
        "speed_reward": speed_reward,
        "potential_reward": potential_reward
    }

    return total_reward, reward_components
