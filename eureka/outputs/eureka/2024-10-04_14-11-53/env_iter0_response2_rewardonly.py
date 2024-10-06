@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the forward velocity component (assuming y-axis is forward)
    forward_velocity = velocity[:, 0]  # Assuming the first component corresponds to the forward direction

    # Calculate a reward based on forward velocity
    reward_forward_velocity = torch.mean(forward_velocity)

    # Calculate a penalty based on the distance traveled
    distance_traveled = torch.norm(torso_position, p=2, dim=-1)
    reward_distance = -distance_traveled / dt
    
    # Together, these two components will encourage the ant to run forward as fast as possible
    total_reward = reward_forward_velocity + reward_distance

    # Define temperature parameters for reward transformations
    temperature_forward = 0.1
    temperature_distance = 0.1

    # Transforming the rewards to have a fixed range
    transformed_reward_forward = torch.exp(temperature_forward * reward_forward_velocity)
    transformed_reward_distance = torch.exp(temperature_distance * reward_distance)

    # Combine transformed rewards
    total_transformed_reward = transformed_reward_forward + transformed_reward_distance

    # Individual reward components
    reward_components = {
        "forward_velocity": transformed_reward_forward,
        "distance_penalty": transformed_reward_distance
    }

    return total_transformed_reward, reward_components
