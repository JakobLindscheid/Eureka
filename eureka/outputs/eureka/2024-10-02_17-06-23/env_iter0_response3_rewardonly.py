@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameter for reward transformation
    speed_temp = 0.1  
    direction_temp = 0.1

    # Compute the reward components
    forward_speed = velocity[:, 0]  # Assuming that the forward velocity is the first component of velocity
    distance_to_target = torch.norm(torso_position - target, p=2, dim=-1)

    # Normalized rewards
    speed_reward = torch.exp(forward_speed / speed_temp)
    direction_reward = torch.exp(-distance_to_target / direction_temp)

    # Total reward
    total_reward = speed_reward + direction_reward

    # Create reward components dictionary
    reward_components = {
        "speed_reward": speed_reward,
        "direction_reward": direction_reward,
    }

    return total_reward, reward_components
