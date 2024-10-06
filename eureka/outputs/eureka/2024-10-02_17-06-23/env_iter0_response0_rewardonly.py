@torch.jit.script
def compute_reward(torso_position: torch.Tensor, velocity: torch.Tensor, goal_pos: torch.Tensor,
                   dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define the temperature parameters for the transformed rewards
    speed_temperature = 0.1
    goal_distance_temperature = 0.1

    # Compute the reward for running speed
    speed_reward = torch.norm(velocity[:, :2], p=2, dim=-1)  # Reward based on forward speed (X and Y plane)
    transformed_speed_reward = torch.exp(speed_temperature * speed_reward)

    # Compute the reward for goal proximity
    to_goal = goal_pos - torso_position
    distance_to_goal = torch.norm(to_goal, p=2, dim=-1)
    goal_reward = -distance_to_goal  # We want to minimize the distance to the goal
    transformed_goal_reward = torch.exp(goal_distance_temperature * goal_reward)

    # Total reward is the sum of speed and goal rewards
    total_reward = transformed_speed_reward + transformed_goal_reward

    # Create a dictionary for individual components
    reward_components = {
        "speed_reward": transformed_speed_reward,
        "goal_reward": transformed_goal_reward
    }

    return total_reward, reward_components
