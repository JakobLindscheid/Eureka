@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for forward velocity using a higher temperature to encourage distinction at higher speeds
    temp_velocity = 0.1
    reward_velocity = torch.exp(temp_velocity * forward_velocity) - 1

    # Penalize high joint forces more significantly
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    temp_force_penalty = 0.05
    penalty_force = torch.exp(-temp_force_penalty * force_magnitude)

    # Total reward is the combination of velocity reward and force penalty with adjusted scaling
    total_reward = reward_velocity * penalty_force

    # Creating a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }

    return total_reward, reward_dict
