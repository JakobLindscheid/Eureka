@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction (forward velocity)
    velocity = root_states[:, 7:10]  # Assuming these correspond to [vx, vy, vz]
    forward_velocity = velocity[:, 0]  # Forward movement in the x direction
    
    # Temperature parameters for reward transformation
    running_temp = 0.5  # Adjust this to scale forward velocity reward
    action_temp = 0.8   # Adjust this to scale action penalty

    # Reward for running: scale and apply an exponential transformation for sensitivity
    running_reward = torch.exp(running_temp * forward_velocity) - 1  # Exponential to generate more positive values

    # Stronger penalty for excessive actions (use squared norm for punitive measurement)
    action_penalty = torch.exp(action_temp * -torch.norm(actions, p=2, dim=-1)) - 1  # Squared to discourage erratic actions

    # Total reward is a sum of both components, normalized within a fixed range
    total_reward = torch.clip(running_reward + action_penalty, min=-10.0, max=10.0)

    # Reward components dictionary for monitoring
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }

    return total_reward, reward_components
