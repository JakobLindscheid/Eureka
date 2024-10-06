@torch.jit.script
def compute_reward(torso_velocity: torch.Tensor, target_position: torch.Tensor, temperature_velocity: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the forward motion vector
    forward_velocity = torso_velocity[:, 0]  # Getting the x-component of velocity assuming forward is along x-axis
    distance_to_target = torch.norm(target_position - torso_velocity[:, :3], p=2, dim=1)

    # Reward for speed - we use an exponential function to normalize the velocity reward
    speed_reward = torch.exp(forward_velocity * temperature_velocity)
    
    # Penalizing distance to target to encourage reaching it
    distance_penalty = -torch.exp(-distance_to_target / temperature_velocity)
    
    # Total reward is a combination of speed reward and distance penalty
    total_reward = speed_reward + distance_penalty

    # Create a dictionary for individual components
    reward_components = {
        'speed_reward': speed_reward,
        'distance_penalty': distance_penalty
    }
    
    return total_reward, reward_components
