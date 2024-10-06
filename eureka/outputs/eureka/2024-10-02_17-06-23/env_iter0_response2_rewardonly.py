@torch.jit.script
def compute_reward(object_pos: torch.Tensor, goal_pos: torch.Tensor, 
                   dof_vel: torch.Tensor, heading_vec: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    speed_scale = 1.0
    orientation_scale = 0.5
    temperature_speed = 0.1
    temperature_orientation = 0.1

    # Compute speed of the ant
    speed = torch.norm(dof_vel, p=2, dim=-1)

    # Compute the desired heading towards the goal
    to_goal = goal_pos - object_pos
    to_goal[:, 2] = 0.0  # Ignore vertical component for heading

    desired_heading = torch.nn.functional.normalize(to_goal, p=2, dim=-1)
    heading_dot_product = torch.sum(desired_heading * heading_vec, dim=-1)

    # Speed reward
    speed_reward = speed * speed_scale
    
    # Heading alignment reward
    orientation_reward = heading_dot_product * orientation_scale
    
    # Total reward
    total_reward = torch.exp(speed_reward / temperature_speed) + torch.exp(orientation_reward / temperature_orientation)

    # Individual reward components
    reward_components = {
        'speed_reward': speed_reward,
        'orientation_reward': orientation_reward
    }

    return total_reward, reward_components
