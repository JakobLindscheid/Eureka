@torch.jit.script
def compute_reward(self, dof_vel: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Hyperparameters for normalization
    speed_temperature = 0.1  # Adjust this to scale the speed reward

    # Calculate the speed (magnitude of the velocity)
    speed = torch.norm(dof_vel[:, 0:2], p=2, dim=-1)  # Consider only the x and y components for running

    # Calculate reward based on speed
    speed_reward = torch.exp(speed / speed_temperature)

    # Total reward is the speed reward
    total_reward = speed_reward

    # Create a dictionary for individual reward components
    reward_components = {
        "speed_reward": speed_reward,
    }

    return total_reward, reward_components
