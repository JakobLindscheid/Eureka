@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, target: torch.Tensor, 
                   dt: float, max_velocity: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward transformations
    temp_pos = 0.1
    temp_vel = 0.1

    # Compute distance from the torso position to the target
    distance_to_target = torch.norm(torso_position - target, p=2)
    normalized_distance = distance_to_target / torch.norm(target)

    # Reward for getting closer to the target, negative to encourage distance reduction
    position_reward = -normalized_distance

    # Compute forward velocity (consider only x-axis for forward running)
    forward_velocity = velocity[:, 0]
    # Normalize the velocity for reward purpose
    normalized_velocity = forward_velocity / max_velocity

    # Reward for maximizing forward velocity
    velocity_reward = normalized_velocity

    # Total reward combining position and velocity rewards
    total_reward = torch.exp(position_reward / temp_pos) + torch.exp(velocity_reward / temp_vel)

    # Individual reward components
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward
    }

    return total_reward, reward_components
