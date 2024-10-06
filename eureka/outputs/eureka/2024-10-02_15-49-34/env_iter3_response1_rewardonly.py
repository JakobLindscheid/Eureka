@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target: torch.Tensor, 
                   dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperatures for reward components
    speed_temp = 0.1
    distance_temp = 0.05

    # Reward for speed: encourage the ant to move forward (in the direction of the target)
    speed_reward = velocity[:, 0]  # Forward speed component
    speed_reward = torch.exp(speed_temp * speed_reward)

    # Reward for distance: penalize distance to the target
    to_target = target - torso_position
    to_target[:, 2] = 0.0  # Project to the horizontal plane
    distance_reward = -torch.norm(to_target, p=2, dim=-1)  # Closer distance yields higher reward
    distance_reward = torch.exp(distance_temp * distance_reward)

    # Total reward
    total_reward = speed_reward + distance_reward

    # Create reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'distance_reward': distance_reward
    }

    return total_reward, reward_components
