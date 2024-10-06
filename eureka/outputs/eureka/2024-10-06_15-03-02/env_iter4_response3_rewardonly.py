@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the forward direction (x-axis)
    torso_velocity = root_states[:, 7:10]  # Assuming torso's velocity is located here
    forward_velocity = torso_velocity[:, 0]  # x component (forward direction)
    
    # Define a temperature variable for reward scaling
    temperature = 0.1
    
    # Compute the individual reward component: forward velocity
    forward_reward = forward_velocity * forward_velocity  # Squared for more emphasis
    normalized_forward_reward = torch.exp(forward_reward / temperature)

    # Total reward is the normalized sum of all reward components
    total_reward = normalized_forward_reward
    
    # Create a dictionary to return individual reward components
    reward_components = {
        'forward_reward': normalized_forward_reward
    }

    return total_reward, reward_components
