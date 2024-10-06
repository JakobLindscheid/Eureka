@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target: torch.Tensor, 
                   up_vec: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for reward components
    speed_temp = 0.1
    height_temp = 0.1

    # Calculate the forward speed component (along the x-axis)
    forward_speed = velocity[:, 0]
    speed_reward = torch.exp(forward_speed / speed_temp)

    # Calculate the height of the torso above a reference plane (e.g., z=0)
    height = torso_position[:, 2]
    height_reward = torch.exp(height / height_temp)

    # Calculate distance to the target position
    to_target = target - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)
    distance_reward = -distance_to_target  # Reward for being closer to the target

    # Combine rewards
    total_reward = speed_reward + height_reward + distance_reward

    # Create a dictionary for individual reward components
    reward_components = {
        'speed_reward': speed_reward,
        'height_reward': height_reward,
        'distance_reward': distance_reward
    }

    return total_reward, reward_components
