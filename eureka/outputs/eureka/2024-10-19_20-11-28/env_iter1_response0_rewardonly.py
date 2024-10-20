@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity (assuming the target direction is along the x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Normalize forward velocity using an exponential transformation
    velocity_temp = 0.1
    reward_velocity = torch.exp(velocity_temp * forward_velocity)
    
    # Penalize high joint forces with a transformed penalty
    force_magnitude = torch.norm(dof_force_tensor, p=2, dim=1)
    force_temp = 0.05
    penalty_force = torch.exp(-force_temp * force_magnitude)
    
    # Combine rewards and penalties
    total_reward = reward_velocity * penalty_force
    
    # Reward dictionary for analyzing components
    reward_dict = {
        "reward_velocity": reward_velocity,
        "penalty_force": penalty_force
    }
    
    return total_reward, reward_dict
