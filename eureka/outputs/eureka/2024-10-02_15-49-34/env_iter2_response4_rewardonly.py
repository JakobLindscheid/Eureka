@torch.jit.script
def compute_reward(torso_position: torch.Tensor, torso_velocity: torch.Tensor, 
                   targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    forward_direction = torch.tensor([1.0, 0.0, 0.0], device=torso_position.device)  # Assuming forward is in the x direction
    target_direction = targets - torso_position
    target_direction[:, 2] = 0.0  # Ignore the vertical component
    
    # Normalize target direction
    target_norm = torch.norm(target_direction, p=2, dim=1, keepdim=True)
    target_direction_normalized = target_direction / (target_norm + 1e-8)

    # Reward for moving towards the target
    forward_reward = torch.sum(torso_velocity * target_direction_normalized, dim=1)
    
    # Penalty for staying still
    stillness_penalty = -torch.norm(torso_velocity, dim=1)

    # Total reward
    total_reward = forward_reward + stillness_penalty

    # Normalize the reward components
    forward_reward_temp = 0.1
    stillness_penalty_temp = 0.1

    normalized_forward_reward = torch.exp(forward_reward / forward_reward_temp)
    normalized_stillness_penalty = torch.exp(stillness_penalty / stillness_penalty_temp)

    # Final total reward transformation
    total_reward = normalized_forward_reward + normalized_stillness_penalty
    
    reward_components = {
        'forward_reward': normalized_forward_reward,
        'stillness_penalty': normalized_stillness_penalty
    }

    return total_reward, reward_components
