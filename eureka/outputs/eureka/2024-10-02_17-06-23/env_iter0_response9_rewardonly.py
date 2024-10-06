@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, 
                   targets: torch.Tensor, contact_force: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature values for reward transformation
    temp_speed = 0.1
    temp_contact = 0.1

    # Calculate the distance to the target's X position (projected on the X-axis)
    distance_to_target = targets[:, 0] - torso_position[:, 0]
    
    # Compute the forward speed by projecting the velocity onto the X-axis
    forward_speed = velocity[:, 0]
    
    # Reward for moving towards the target
    reward_move_towards = distance_to_target * 1.0  # Weight towards reaching target
    
    # Reward for speed (encouraging faster movement)
    reward_speed = forward_speed
    # Apply transformation for normalization
    reward_speed = torch.exp(temp_speed * reward_speed)

    # Reward for contact force (to encourage stability and avoid falling)
    total_contact_force = torch.norm(contact_force, p=2, dim=-1)
    reward_contact = -total_contact_force  # Negative because we want lesser contact forces
    # Apply transformation for normalization
    reward_contact = torch.exp(temp_contact * reward_contact)

    # Combine rewards
    total_reward = reward_move_towards + reward_speed + reward_contact
    
    # Store individual rewards in a dictionary
    individual_rewards = {
        'reward_move_towards': reward_move_towards,
        'reward_speed': reward_speed,
        'reward_contact': reward_contact,
    }

    return total_reward, individual_rewards
