class Generator(nn.Module):
    """Maps a latent vector to a point in data space.

    Same two hidden layers as the hand-written version, and the same $\\tanh$ by default,
    so that the two can be compared step by step. The `activation` argument is there
    because the two-dimensional section of this part will want `LeakyReLU`, which is the
    usual choice in a GAN; the $\\tanh$ is a concession to the plain gradient descent of
    the one-dimensional section, which tolerates a large step size only on a smooth
    network.

    The output layer is a plain `nn.Linear`: the toy data is not bounded, so nothing
    should squash the output here. On MNIST, in Part II, that last layer will get a
    `tanh`.
    """

    def __init__(self, latent_dim=1, data_dim=1, hidden=64, activation=None):
        super().__init__()
        self.latent_dim = latent_dim
        activation = activation if activation is not None else nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), activation,
            nn.Linear(hidden, hidden), activation,
            nn.Linear(hidden, data_dim),
        )

    def forward(self, z):
        return self.net(z)

    def sample(self, n, device=device):
        """Draw n new points: latent noise in, data out."""
        return self.forward(torch.randn(n, self.latent_dim, device=device))
