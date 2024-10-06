@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso velocity from the root states
    velocity = root_states[:, 7:10]  # Assuming the velocity is the 7th to 9th elements
    torso_position = root_states[:, 0:3]  # Assuming the torso position is the first three elements

    # Calculate the forward direction (assuming forward is along the x-axis, index 0)
    forward_velocity = velocity[:, 0]

    # Define temperature for reward normalization
    temp_forward = 1.0

    # Compute rewards
    reward_forward = torch.exp(forward_velocity / temp_forward)

    # Calculate distance towards the target
    to_target = targets - torso_position
    distance_to_target = torch.norm(to_target, p=2, dim=-1)

    # We can incentivize moving towards the target negatively
    reward_distance = -distance_to_target / dt  # Encourage shorter distance to the target

    # Overall reward is a combination of moving forward and getting closer to the target
    total_reward = reward_forward + reward_distance

    # Return the total reward and individual reward components
    return total_reward, {"reward_forward": reward_forward, "reward_distance": reward_distance}
