@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the velocity of the torso
    velocity = root_states[:, 7:10]  # Torso velocity

    # Compute speed as the L2 norm of the velocity vector
    speed = torch.norm(velocity, p=2, dim=-1)

    # Define a temperature parameter for reward transformation
    speed_temp = 0.1

    # Transform speed to normalize reward
    normalized_speed = torch.exp(speed_temp * speed)

    # Define the total reward as the normalized speed
    total_reward = normalized_speed

    # Create a dictionary for individual reward components
    reward_components = {'speed': normalized_speed}

    return total_reward, reward_components
