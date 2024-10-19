@torch.jit.script
def compute_reward(obs_buf: torch.Tensor, velocity: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Hyperparameters
    speed_weight: float = 1.0  # Weight for speed reward
    action_weight: float = -0.01  # Weight for action penalty to encourage efficiency
    temperature_speed: float = 0.5  # Temperature for speed reward transformation
    temperature_action: float = 0.1  # Temperature for action penalty transformation
    
    # Compute speed reward, encouraging higher forward velocity (velocity in z-direction)
    forward_velocity = velocity[:, 2]  # Assuming forward velocity is in the z-axis
    speed_reward = torch.mean(forward_velocity)  # Mean speed across all humanoid agents
    
    # Apply transformation to normalize the reward
    transformed_speed_reward = torch.exp(speed_weight * speed_reward / temperature_speed)
    
    # Compute action penalty (encouraging less excessive actions)
    action_penalty = torch.mean(torch.norm(action, p=2, dim=-1))  # L2 norm of actions
    transformed_action_penalty = torch.exp(action_weight * action_penalty / temperature_action)

    # Total reward is speed reward adjusted by action penalty
    total_reward = transformed_speed_reward - transformed_action_penalty
    
    # Individual components dictionary
    reward_components = {
        "speed_reward": transformed_speed_reward,
        "action_penalty": transformed_action_penalty,
    }
    
    return total_reward, reward_components
