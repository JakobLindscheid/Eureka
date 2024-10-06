@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for normalization
    speed_temp = 0.5

    # Extract the torso velocity (x, y, z components)
    torso_velocity = root_states[:, 7:10]  # Shape: (N, 3)
    
    # Calculate the speed of the torso (magnitude of the velocity vector)
    speed = torch.norm(torso_velocity, p=2, dim=-1)  # Shape: (N,)

    # Scaling the reward based on speed to encourage faster running
    reward = speed / (1.0 + speed)  # Use a sigmoid-like function to keep rewards in a manageable range 

    # Create individual reward components for monitoring
    reward_components = {
        'speed': speed,
    }
    
    return reward.sum(), reward_components  # Return total reward and individual components
