@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, goal_pos: torch.Tensor, up_vec: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants for reward shaping
    forward_direction = torch.tensor([1.0, 0.0, 0.0], device=torso_position.device)  # Assuming +x direction is forward
    velocity_reward_temp = 0.5  # Temperature for velocity reward transformation
    position_reward_temp = 0.5  # Temperature for position reward transformation

    # Compute the forward velocity component (along x-axis)
    forward_velocity = velocity @ forward_direction  # Dot product with forward direction

    # Reward for moving forward
    velocity_reward = torch.exp(forward_velocity * velocity_reward_temp)

    # Reward for being closer to the goal position
    distance_to_goal = torch.norm(goal_pos - torso_position, p=2, dim=-1)
    position_reward = torch.exp(-(distance_to_goal / dt) * position_reward_temp)
    
    # Total reward combining both components
    total_reward = velocity_reward + position_reward

    # Construct reward components dictionary
    reward_components = {
        'velocity_reward': velocity_reward,
        'position_reward': position_reward
    }

    return total_reward, reward_components
