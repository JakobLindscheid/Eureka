@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target_position: torch.Tensor, 
                   prev_potentials: torch.Tensor, current_time: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Parameters
    desired_heading = torch.tensor([1.0, 0.0, 0.0], device=torso_position.device)  # Forward direction
    forward_speed_weight: float = 1.0  # Weight for forward speed reward
    tilt_penalty_weight: float = 0.1    # Weight for tilt penalty
    temperature: float = 0.01            # Temperature for reward transformation

    # Calculate the heading and speed of the torso
    velocity_norm = torch.norm(velocity, p=2, dim=-1)
    forward_speed = torch.dot(velocity, desired_heading)

    # Reward components
    forward_speed_reward = forward_speed_weight * forward_speed
    tilt_penalty = tilt_penalty_weight * torch.abs(torso_position[:, 2])  # Assuming z-axis is the "tilt"

    # Total reward
    total_reward = forward_speed_reward - tilt_penalty
    
    # Applying temperature transformation
    total_reward = torch.exp(total_reward / temperature)

    # Store individual rewards
    reward_components = {
        "forward_speed_reward": forward_speed_reward,
        "tilt_penalty": -tilt_penalty  # Negative because we want less tilt
    }

    return total_reward, reward_components
