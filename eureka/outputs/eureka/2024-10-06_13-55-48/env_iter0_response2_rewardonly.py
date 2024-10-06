@torch.jit.script
def compute_reward(
    torso_position: torch.Tensor, 
    torso_velocity: torch.Tensor,
    target_position: torch.Tensor,
    previous_potential: torch.Tensor,
    current_potential: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature variable for reward normalization
    speed_temperature = 1.0
    potential_temperature = 1.0

    # Calculate speed towards the target
    speed = torch.norm(torso_velocity, p=2, dim=-1)

    # Reward based on speed; the faster the humanoid runs, the higher the reward
    speed_reward = speed / (1.0 + speed_temperature)  # Normalizing speed

    # Reward based on improvement in potential (encouraging running towards the target)
    potential_improvement = current_potential - previous_potential
    potential_reward = potential_improvement.clone()
    potential_reward[potential_reward < 0] = 0  # Discard negative improvements

    # Combine rewards
    total_reward = speed_reward + potential_reward / (1.0 + potential_temperature)  # Normalizing potential reward

    # Create a dictionary of individual rewards
    reward_components = {
        "speed_reward": speed_reward,
        "potential_reward": potential_reward,
        "total_reward": total_reward
    }

    return total_reward, reward_components
