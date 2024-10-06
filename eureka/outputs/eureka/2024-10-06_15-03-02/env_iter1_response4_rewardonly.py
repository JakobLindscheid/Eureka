@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward transformation
    speed_temp = 0.5

    # Extract the velocity of the torso (the first three elements of root_states)
    torso_velocity = root_states[:, 7:10]  # Assuming the 7th to 9th indices are the velocity
        
    # Compute the speed (magnitude of the velocity)
    speed = torch.norm(torso_velocity, p=2, dim=-1)

    # Normalize and transform the speed reward
    speed_reward = torch.exp(speed / speed_temp)

    total_reward = speed_reward

    # Collect individual rewards
    reward_components = {
        "speed_reward": speed_reward,
    }

    return total_reward, reward_components
