@torch.jit.script
def compute_reward(torso_position: torch.Tensor, torso_velocity: torch.Tensor, angle_to_target: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters for transforming rewards
    velocity_temp = 0.1
    angle_temp = 0.1
    
    # Calculate the forward velocity component (assuming forward is along the x-axis)
    forward_velocity = torso_velocity[:, 0]

    # Calculate distance to the target (projected movement)
    distance_to_target = torch.norm(targets - torso_position, p=2, dim=-1)

    # Reward for moving in the direction of the target
    movement_reward = torch.exp(forward_velocity / velocity_temp)

    # Penalize the angle to the target (the smaller, the better)
    angle_penalty = -torch.exp(-angle_to_target / angle_temp)

    # Total reward combining movement reward and angle penalty
    total_reward = movement_reward + angle_penalty

    # Reward components
    reward_components = {
        'movement_reward': movement_reward,
        'angle_penalty': angle_penalty
    }

    return total_reward, reward_components
