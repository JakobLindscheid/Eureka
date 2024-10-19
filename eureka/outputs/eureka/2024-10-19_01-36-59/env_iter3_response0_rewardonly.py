@torch.jit.script
def compute_reward(root_states: torch.Tensor, dt: float, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack the root states
    torso_velocity = root_states[:, 7:10]  # Velocity components
    forward_velocity = torso_velocity[:, 0]  # Forward velocity (x-axis)
    torso_position = root_states[:, 0:3]
    
    # Calculate distance to the target (2D, ignoring height)
    to_target = targets - torso_position
    distance_to_target = torch.sqrt(to_target[:, 0]**2 + to_target[:, 1]**2)

    # Reward components
    reward_forward_speed = forward_velocity  # Forward speed indicates immediate progress
    sustained_speed = torch.max(forward_velocity, torch.tensor(0.0, device=root_states.device))  # Non-negative forward speed
    
    # Reformulated heading reward based on how close the ant is to heading towards the target
    reward_heading = torch.exp(-distance_to_target / (5.0 + 1e-5))  # Heavily penalizes distance with an exponential function

    # New reward for reaching the target
    reached_target = (distance_to_target < 1.0).float()  # Reward for being within 1 unit of the target
    reward_reach_target = reached_target * 10.0  # Strong reward for reaching the target
    
    # Temperature parameters for scaling rewards
    temperature_forward = 5.0
    temperature_sustained_speed = 5.0
    temperature_heading = 5.0

    # Transforming the rewards for scaling
    transformed_reward_forward_speed = torch.exp(reward_forward_speed / temperature_forward) - 1
    transformed_sustained_speed = torch.exp(sustained_speed / temperature_sustained_speed) - 1  
    transformed_reward_heading = reward_heading  # Keep as is for normalized value
    transformed_reward_reach_target = reward_reach_target  

    # Total reward calculation
    total_reward = (
        transformed_reward_forward_speed + 
        transformed_sustained_speed + 
        transformed_reward_heading + 
        transformed_reward_reach_target
    )

    # Create a dictionary for the individual reward components
    reward_components = {
        'reward_forward_speed': transformed_reward_forward_speed,
        'sustained_speed': transformed_sustained_speed,
        'reward_heading': transformed_reward_heading,
        'reward_reach_target': transformed_reward_reach_target,
    }

    return total_reward, reward_components
