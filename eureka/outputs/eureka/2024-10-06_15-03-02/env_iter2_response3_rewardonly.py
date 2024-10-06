@torch.jit.script
def compute_reward(vel_loc: torch.Tensor, prev_potentials: torch.Tensor, potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature variables for reward normalization
    speed_temp = 0.1
    potential_temp = 0.05
    
    # Calculate the running speed reward based on the x-component (forward speed)
    speed_reward = vel_loc[:, 0]  # Assuming that the first component represents forward velocity
    
    # Calculate potential change as a reward component (negative potential indicates closer to the target)
    potential_reward = potentials - prev_potentials
    
    # Total reward that encourages faster running and reduces potentials
    total_reward = torch.exp(speed_temp * speed_reward) + torch.exp(potential_temp * potential_reward)
    
    # Reward components dictionary
    reward_components = {
        "speed_reward": speed_reward,
        "potential_reward": potential_reward,
    }
    
    return total_reward, reward_components
