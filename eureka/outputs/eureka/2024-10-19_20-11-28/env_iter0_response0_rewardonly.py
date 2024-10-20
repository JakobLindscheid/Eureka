@torch.jit.script
def compute_reward(velocity: torch.Tensor, angle_to_target: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward weight parameters (can be fine-tuned)
    velocity_weight = 1.0
    stability_weight = 0.1
    energy_consumption_weight = 0.01

    # Temperature parameters for reward transformations
    velocity_temp = 1.0
    stability_temp = 1.0

    # Calculate velocity reward: encourage higher velocities
    velocity_reward = velocity[:, 0]  # Assuming the first component is the forward velocity
    velocity_reward = torch.exp(velocity_temp * velocity_reward) - 1.0

    # Calculate stability reward: discourage large deviation in angle to the target
    stability_reward = -torch.abs(angle_to_target)
    stability_reward = torch.exp(stability_temp * stability_reward) - 1.0

    # Calculate energy consumption penalty: discourage large actions
    energy_consumption_penalty = -torch.mean(actions.pow(2), dim=-1)

    # Total reward
    total_reward = (velocity_weight * velocity_reward +
                    stability_weight * stability_reward +
                    energy_consumption_weight * energy_consumption_penalty)

    # Compile reward components into a dictionary
    reward_components = {
        "velocity_reward": velocity_weight * velocity_reward,
        "stability_reward": stability_weight * stability_reward,
        "energy_consumption_penalty": energy_consumption_weight * energy_consumption_penalty
    }

    return total_reward, reward_components
