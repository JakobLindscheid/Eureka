class Ant(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self, obs, actions):
        self.actions = torch.tensor(actions)
        self.x_positions = torch.tensor(x_position)
        self.y_positions = torch.tensor(y_position)
        self.z_positions = torch.tensor(obs[:, 0])
        self.x_velocities = torch.tensor(obs[:, 13])
        self.y_velocities = torch.tensor(obs[:, 14])
        self.z_velocities = torch.tensor(obs[:, 15])     