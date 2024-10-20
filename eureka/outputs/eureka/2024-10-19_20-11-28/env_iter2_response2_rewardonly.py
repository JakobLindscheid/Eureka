@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   dof_force_tensor: torch.Tensor,
                   torso_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Extract forward velocity (assuming x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for achieving higher forward velocity - amplify effect
    velocity_temperature = 0.2  # Increased temperature for sharper reward curve
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0
    
    # Penalize excessive joint forces
    force_temperature = 0.1  # Adjusted to increase the range of penalization effect
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # Reward for maintaining height (e.g., above a threshold to encourage upright posture)
    height_threshold = 1.0  # Arbitrary threshold for stable running, specific to environment scale
    reward_height = torch.where(torso_position[:, 2] > height_threshold, 1.0, 0.0)

    # Total reward is a combination of velocity, posture penalization, and height bonus
    total_reward = reward_velocity * penalty_force + reward_height

    # Dictionary for debugging purposes
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force,
        "reward_height": reward_height
    }

    return total_reward, reward_dict
