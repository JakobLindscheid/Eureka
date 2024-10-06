@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary information from root_states
    torso_position = root_states[:, 0:3]  # Position (x, y, z)
    velocity = root_states[:, 7:10]        # Velocity (vx, vy, vz)

    # Calculate the forward movement speed in the x direction
    forward_speed = velocity[:, 0]

    # Calculate the distance to target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)

    # Rewards for movement and distance to target
    forward_reward = torch.exp(forward_speed) - 1  # Promotes faster forward speed
    distance_reward = torch.clamp(1.0 - (distance_to_target / 10.0), min=0)  # Reward for coming closer

    # Total reward considering forward movement and distance reduction
    total_reward = forward_reward + distance_reward

    # Define individual reward components
    reward_components = {
        'forward_reward': forward_reward,
        'distance_reward': distance_reward
    }

    return total_reward, reward_components
