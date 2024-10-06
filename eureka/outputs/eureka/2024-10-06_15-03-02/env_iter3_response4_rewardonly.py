@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso velocity from the root states
    torso_velocity = root_states[:, 7:10]  # assuming the velocity is in the 7th to 9th elements
    
    # Calculate the forward speed of the humanoid (use the z-axis direction)
    forward_speed = torso_velocity[:, 0]  # taking the x-component as the forward speed

    # Define a temperature parameter to control the reward sensitivity
    temperature_speed = 1.0  # you can adjust this value as needed

    # Normalize the forward speed reward using the exponential transformation
    speed_reward = torch.exp(forward_speed / temperature_speed)
    
    # Total reward is the speed reward
    total_reward = speed_reward

    # Create a reward components dictionary
    reward_components = {
        'speed_reward': speed_reward,
    }

    return total_reward, reward_components
