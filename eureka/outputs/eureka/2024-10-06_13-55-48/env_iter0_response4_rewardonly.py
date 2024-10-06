@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso velocity from the root_states
    torso_velocity = root_states[:, 7:10]  # linear velocity: x, y, z

    # Calculate the speed of the humanoid as the norm of the velocity vector
    speed = torch.norm(torso_velocity, p=2, dim=-1)

    # Define a temperature parameter for reward scaling
    temperature = 0.5

    # Transform the speed into a reward component and normalize it
    reward_speed = torch.exp(speed / temperature)

    # Define a reward for getting closer to the target
    to_target = targets - root_states[:, 0:3]  # torso position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)

    # Transform the distance to target into a reward (negative because we want to minimize it)
    reward_distance = torch.exp(-distance_to_target / temperature)

    # Total reward is a combination of speed and distance rewards
    total_reward = reward_speed + reward_distance

    # Create a dictionary to hold individual reward components
    reward_components = {
        'speed_reward': reward_speed,
        'distance_reward': reward_distance,
    }

    return total_reward, reward_components
