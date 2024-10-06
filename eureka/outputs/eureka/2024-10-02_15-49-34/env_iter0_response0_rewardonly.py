@torch.jit.script
def compute_reward(torso_position: torch.Tensor, 
                   velocity: torch.Tensor, 
                   targets: torch.Tensor,
                   dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward transformations
    velocity_temp = 1.0  # Temperature for velocity reward
    distance_temp = 1.0  # Temperature for distance reward
    
    # Reward for forward velocity
    forward_velocity = velocity[:, 0]  # only the x-component of the velocity
    velocity_reward = torch.exp(torch.mean(forward_velocity) / velocity_temp)

    # Reward for distance to the target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # ignore the z-component
    distance_to_target = torch.norm(to_target, p=2, dim=-1)
    distance_reward = torch.exp(-distance_to_target / distance_temp)

    # Total reward as a combination of velocity and distance rewards
    total_reward = velocity_reward + distance_reward

    # Reward components dictionary
    reward_components = {
        'velocity_reward': velocity_reward,
        'distance_reward': distance_reward
    }

    return total_reward, reward_components
