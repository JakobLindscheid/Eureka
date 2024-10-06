@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dt: float, distance_traveled: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis
    
    # Reward for running forward: positive reward for higher speed
    running_reward_temp = 5.0  # Temperature for the running reward
    running_reward = torch.exp(forward_velocity / running_reward_temp)  # Scale to make it more sensitive
    
    # Modified Action Penalty: penalizing only drastic changes
    action_change = torch.norm(actions, p=2, dim=-1)  # Calculate action change
    action_penalty_temp = 5.0  # Temperature for action penalty
    action_penalty = -torch.exp(action_change / action_penalty_temp)  # Penalty for action changes
    
    # Reward for distance traveled
    distance_temp = 10.0  # Temperature for distance reward
    distance_reward = torch.exp(distance_traveled / distance_temp)  # Incentivize longer runs
    
    # Calculate total reward
    total_reward = running_reward + action_penalty + distance_reward
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty,
        "distance_reward": distance_reward
    }
    
    return total_reward, reward_components
