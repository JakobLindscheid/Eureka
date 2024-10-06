@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target_pos: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature variables to control the reward component scaling
    speed_temp = 0.1
    tilt_temp = 0.1
    
    # Compute forward speed as the projection of the velocity onto the forward direction
    forward_direction = target_pos - torso_position
    forward_direction[:, 2] = 0.0  # Ignore the vertical component
    forward_direction = torch.nn.functional.normalize(forward_direction, dim=-1)
    
    speed = torch.sum(velocity * forward_direction, dim=-1)
    
    # Reward for forward speed, we apply a transformation to normalize it
    speed_reward = torch.exp(speed * speed_temp)
    
    # Compute the penalty based on the tilt of the plane (using up vector)
    tilt_penalty = torch.norm(up_vec, p=2, dim=-1)  # Assume we want to minimize the tilt from vertical
    tilt_reward = -torch.exp(tilt_penalty * tilt_temp)
    
    # Total reward is the combination of speed reward and tilt penalty
    total_reward = speed_reward + tilt_reward
    
    # Package individual rewards to return
    reward_components = {
        'speed_reward': speed_reward,
        'tilt_reward': tilt_reward
    }
    
    return total_reward, reward_components
