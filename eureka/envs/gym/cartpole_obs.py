class CartPole(VectorEnv):
    """Rest of the environment definition omitted."""
    def compute_observations(self, action):
        self.cart_position = torch.tensor(obs[:, 0])
        self.cart_velocity = torch.tensor(obs[:, 1])
        self.pole_angle = torch.tensor(obs[:, 2])
        self.pole_velocity = torch.tensor(obs[:, 3])