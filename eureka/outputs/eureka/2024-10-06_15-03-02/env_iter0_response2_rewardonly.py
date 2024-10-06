@torch.jit.script
def compute_reward(velocity: torch.Tensor, target_velocity: float, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    speed_scale = 1.0
    velocity_temp = 0.1  # Temperature for velocity reward transformation
    
    # Calculate the speed of the humanoid
    speed = torch.norm(velocity, p=2, dim=-1)  # Compute the speed as the L2 norm of the velocity vector
    
    # Reward based on speed towards the target velocity
    speed_reward = speed - target_velocity  # Reward is the excess speed over the desired target velocity
    speed_reward_transformed = torch.exp(speed_scale * speed_reward)  # Transforming the speed reward
    
    # Total reward is just the transformed speed reward
    total_reward = speed_reward_transformed / dt  # Normalize by delta time for consistency
    
    return total_reward, {"speed_reward": speed_reward_transformed}
