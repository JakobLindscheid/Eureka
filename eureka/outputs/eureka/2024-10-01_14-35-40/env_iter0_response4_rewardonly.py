@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Calculate the forward velocity (i.e., along the x-axis)
    forward_velocity = velocity[:, 0]

    # Reward based on forward speed: higher speed yields higher reward
    reward_forward_velocity = forward_velocity / dt  # Normalizing by time step
    
    # Using a temperature variable for reward transformation
    temp_forward = 0.1  # Temperature for forward velocity reward
    transformed_reward_forward = torch.exp(reward_forward_velocity / temp_forward)

    # Calculate the distance to the target (if needed for shaping)
    distance_to_target = torch.norm(torso_position - targets, p=2, dim=1)

    # Combine rewards: encouraging speed while limiting distance to target
    # Reward shaping can also be done here
    reward = transformed_reward_forward - distance_to_target

    # Collecting individual components of the reward
    reward_components = {
        'forward_velocity_reward': transformed_reward_forward,
        'distance_to_target_penalty': -distance_to_target
    }

    return reward, reward_components
