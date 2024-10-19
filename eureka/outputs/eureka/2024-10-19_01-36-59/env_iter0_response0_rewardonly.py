@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Variables for reward components
    forward_speed = root_states[:, 7]  # Assuming forward speed is represented by the second component of the velocity
    distance_to_target = torch.norm(root_states[:, 0:3] - root_states[:, 3:6], p=2, dim=-1)  # Hypothetical target position
    velocity_reward = forward_speed / dt  # Reward for speed normalized by time step

    # Define temperature parameters
    vel_temp = 0.1  # Adjust this value to control sensitivity of the speed reward

    # Compute the total reward
    total_reward = torch.exp(velocity_reward / vel_temp) - 1.0  # Exponential transformation

    # Collecting individual rewards
    rewards_dict = {
        "forward_speed_reward": velocity_reward,
        "distance_to_target": distance_to_target
    }

    return total_reward, rewards_dict
