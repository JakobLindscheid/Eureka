@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor, potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso velocity from root_states
    velocity = root_states[:, 7:10]  # Assuming the velocity is at indices 7:10
    speed = torch.norm(velocity, p=2, dim=-1)

    # Define temperature parameter for reward normalization
    speed_temperature = 0.1
    potential_difference = potentials - prev_potentials

    # Compute rewards
    speed_reward = torch.exp(speed / speed_temperature)  # Reward based on speed
    potential_reward = potential_difference  # Potential gain (should be negative since it's -norm)

    # Total reward
    total_reward = speed_reward + potential_reward

    # Create a dictionary for individual reward components
    reward_components = {
        'speed_reward': speed_reward,
        'potential_reward': potential_reward
    }

    return total_reward, reward_components
