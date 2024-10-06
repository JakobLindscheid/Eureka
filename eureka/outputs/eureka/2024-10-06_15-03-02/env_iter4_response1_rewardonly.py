@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso velocity from the root states
    torso_velocity = root_states[:, 7:10]
    
    # Compute the speed of the torso (Euclidean norm of velocity)
    speed = torch.norm(torso_velocity, p=2, dim=-1)

    # Define a temperature for reward normalization
    temperature = 0.1

    # Compute the reward based on speed (using an exponential transformation for normalization)
    reward_speed = torch.exp(speed / temperature)

    # Compute the distance to the target (projection to the xy-plane)
    to_target = targets - root_states[:, 0:3]
    to_target[:, 2] = 0  # Ignore the z-component
    distance_to_target = torch.norm(to_target, p=2, dim=-1)

    # A penalty for being far from the target (also normalized)
    temperature_penalty = 0.5
    penalty_distance = torch.exp(-distance_to_target / temperature_penalty)

    # Total reward is the speed reward minus the distance penalty
    total_reward = reward_speed + penalty_distance

    # Create a dictionary of individual reward components
    reward_components = {
        'speed_reward': reward_speed,
        'distance_penalty': penalty_distance
    }

    return total_reward, reward_components
