@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Reward for forward velocity with increased sensitivity
    velocity_temperature = 0.25
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # Adjust penalty force to have a greater impact by increasing its sensitivity
    force_temperature = 0.05
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    penalty_force = torch.exp(-force_temperature * force_magnitude)

    # New reward component for maintaining upright posture
    upright_threshold = 0.8
    reward_upright = torch.where(up_vec[:, 2] > upright_threshold, 1.0, 0.0)
    reward_upright_temperature = 0.5
    reward_upright = torch.exp(reward_upright * reward_upright_temperature) - 1.0

    # Combined reward emphasizes velocity with consideration of force usage and posture maintenance
    total_reward = reward_velocity * penalty_force * reward_upright

    # Create a reward dictionary for analysis
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force,
        "reward_upright": reward_upright
    }

    return total_reward, reward_dict
