@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define a temperature variable for scaling the velocity reward
    temperature_velocity = 0.1

    # Extract the velocity of the humanoid
    velocity = root_states[:, 7:10]  # assuming the 8th to 10th elements correspond to x, y, z velocities

    # Calculate the speed (magnitude of the velocity vector)
    speed = torch.norm(velocity, p=2, dim=-1)

    # Normalize the speed using an exponential transformation
    speed_reward = torch.exp(-temperature_velocity * (1 - speed)).mean()

    # Total reward is primarily based on speed
    total_reward = speed_reward

    # Create a dictionary for individual reward components
    reward_components = {
        'speed_reward': speed_reward,
    }

    return total_reward, reward_components
