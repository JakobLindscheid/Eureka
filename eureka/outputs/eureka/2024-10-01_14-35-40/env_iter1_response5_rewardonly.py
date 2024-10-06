@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor,
                   actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # x, y, z velocities
    forward_velocity = velocity[:, 0]  # Using only the x component for forward speed

    # Enhanced reward for running forward: positive reward for achieving above a threshold speed
    target_speed = 2.0  # Desired speed threshold
    running_reward = torch.where(forward_velocity >= target_speed,
                                 torch.exp((forward_velocity - target_speed) / 2.0), 
                                 torch.zeros_like(forward_velocity))
    running_reward = torch.clip(running_reward, min=0.0, max=1.0)  # Normalize to [0, 1]

    # Strengthened penalty for excessive actions to encourage smoother movement
    action_penalty = -torch.norm(actions, p=2, dim=-1)  
    action_penalty = torch.exp(action_penalty / 2.0)  # Adjust temperature for higher sensitivity

    # Calculate total reward
    total_reward = running_reward + action_penalty
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
