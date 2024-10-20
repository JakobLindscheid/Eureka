@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target running direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Adjusted temperature for reward velocity
    velocity_temperature = 0.3  # Reduced for stability
    reward_velocity = torch.exp(forward_velocity * velocity_temperature) - 1.0

    # New approach for penalty force: Monitor variance in forces for smooth movement
    force_variance_temperature = 0.01  # New temperature for variance analysis
    dof_force_variance = torch.var(dof_force_tensor, dim=1)
    penalty_force_variance = torch.exp(-force_variance_temperature * dof_force_variance) - 1.0  # Penalty for high variance

    # Emphasize forward velocity and variance control
    total_reward = reward_velocity + 0.5 * penalty_force_variance  # Penalty adjusted to match reward balance 

    # Create a dictionary of reward components
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force_variance": penalty_force_variance
    }

    return total_reward, reward_dict
