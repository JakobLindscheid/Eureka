@torch.jit.script
def compute_reward(root_states: torch.Tensor, velocity: torch.Tensor, dof_vel: torch.Tensor, dof_force: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the necessary components
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    
    forward_velocity = velocity[:, 0]  # Assuming x-axis is the forward direction

    # Calculate reward components
    # 1. Velocity Reward: Encourage forward velocity
    velocity_reward = forward_velocity

    # 2. Energy Efficiency Reward: Penalize large joint forces to encourage energy efficiency
    energy_efficiency_penalty = torch.sum(torch.abs(dof_force), dim=-1)
    energy_efficiency_reward = -energy_efficiency_penalty

    # 3. Stability Reward: Encourage upright torso by maximizing projection on the global up vector (z-axis)
    upright_reward = torso_rotation[:, 2]  # Assuming z-component of the rotation indicates uprightness

    # 4. Control Effort Penalty: Penalize large control actions to encourage smooth movements
    control_effort_penalty = torch.sum(torch.abs(dof_vel), dim=-1)
    control_effort_reward = -control_effort_penalty

    # Combine the rewards
    total_reward = velocity_reward + 0.1 * energy_efficiency_reward + 0.5 * upright_reward + 0.01 * control_effort_reward

    # Normalize and transform the reward for stability
    velocity_reward_temp = 1.0
    total_reward = torch.exp(total_reward / velocity_reward_temp)

    # Return the reward and individual components
    return total_reward, {
        "velocity_reward": velocity_reward,
        "energy_efficiency_reward": energy_efficiency_reward,
        "upright_reward": upright_reward,
        "control_effort_reward": control_effort_reward
    }
