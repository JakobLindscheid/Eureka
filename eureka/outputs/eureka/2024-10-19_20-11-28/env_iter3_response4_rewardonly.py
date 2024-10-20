@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Adjusting forward velocity reward with a higher temperature
    velocity_temperature = 0.3  # Making the reward for higher speeds more pronounced
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # Revising joint force penalty for a stronger gradient
    force_temperature = 0.1  # Increase magnitude to make them more significant
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # Adding orientation stability incentive, encouraging upright posture
    upright_temperature = 2.0
    upright_penalty = -torch.log(up_vec[:, 2] + 1.0) * upright_temperature

    # Compute total reward
    total_reward = reward_velocity + penalty_force + upright_penalty

    # Reward components dictionary for insight
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force,
        "upright_penalty": upright_penalty,
    }

    return total_reward, reward_dict
