@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for transforming reward components
    speed_temp = 0.2
    alignment_temp = 0.5
    target_reach_temp = 0.1

    # Extract necessary components
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity component, scaled for effective learning
    forward_velocity = velocity[:, 0]

    # Speed reward with enhanced scaling to encourage acceleration
    speed_reward = torch.square(forward_velocity)  # Squared to emphasize faster speeds

    # Calculate distance to target with a reward for getting closer
    direction_to_target = targets - torso_position
    distance_to_target = torch.norm(direction_to_target, p=2, dim=-1) + 1e-6  # Avoid division by zero

    # Target reach reward: provide more meaningful feedback as the agent approaches the target
    target_reach_reward = torch.exp(-distance_to_target / target_reach_temp)

    # Normalize the direction towards the target for alignment
    direction_to_target_normalized = direction_to_target / distance_to_target.unsqueeze(-1)
    alignment_reward = torch.clamp(torch.sum(velocity * direction_to_target_normalized, dim=-1) / (torch.norm(velocity, p=2, dim=-1) + 1e-6), min=0.0)

    # Total reward is the combination of all components
    total_reward = (speed_reward + alignment_reward + target_reach_reward) / 3.0  # Normalize total reward

    # Prepare individual reward components for the return dictionary
    reward_components = {
        'speed_reward': speed_reward,
        'alignment_reward': alignment_reward,
        'target_reach_reward': target_reach_reward,
    }
    
    return total_reward, reward_components
