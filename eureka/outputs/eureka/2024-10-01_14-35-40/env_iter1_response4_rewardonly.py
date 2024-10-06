@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor,
                   actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # x, y, z velocities
    forward_velocity = velocity[:, 0]  # Forward direction
    
    # Stability reward: Penalize based on angular velocity (should not be too high)
    angular_velocity = root_states[:, 10:13]  # Assuming angular velocity is in this slice
    stability_reward = torch.exp(-torch.norm(angular_velocity, p=2, dim=-1) / 5.0)  # More stable motion gets a higher reward
    
    # Combined running reward: Forward velocity favored, but must be stable too
    running_reward = torch.exp(forward_velocity / 10.0) * stability_reward  # Combine both dynamics
    running_reward = torch.clip(running_reward, min=0.0, max=1.0)  # Normalize to [0, 1]

    # Action penalty: Stronger penalty for excessive movement
    action_penalty = -torch.norm(actions, p=2, dim=-1) / 2.0  # Reduce the efficacy of actions
    action_penalty = torch.exp(action_penalty)  # Apply temperature to the penalty

    # Calculate total reward
    total_reward = running_reward + action_penalty
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "stability_reward": stability_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
