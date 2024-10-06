@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack values from the root_states tensor
    torso_position = root_states[:, 0:3]  # (N, 3)
    torso_velocity = root_states[:, 7:10]  # (N, 3)

    # Calculate the forward direction based on the tilt of the plane
    forward_vector = torch.tensor([1.0, 0.0, 0.0], device=root_states.device)  # Assume x-direction is forward
    projected_velocity = torch.dot(torso_velocity, forward_vector)

    # Reward for moving forward; penalize for staying still or moving backward
    forward_reward = torch.clamp(projected_velocity, min=0.0)

    # Reward based on the height of the torso, encouraging the agent to maintain a proper orientation
    height_reward = torch.clamp(torso_position[:, 2], min=0.0)  # Encourage height above the ground

    # Temperature for reward transformation (could tune this)
    forward_temp = 1.0
    height_temp = 1.0

    # Transform rewards to keep them within a fixed range
    total_reward = torch.exp(forward_temp * forward_reward) + torch.exp(height_temp * height_reward)

    # Create a dictionary for individual reward components
    rewards_dict = {
        'forward_reward': forward_reward,
        'height_reward': height_reward,
    }

    return total_reward, rewards_dict
