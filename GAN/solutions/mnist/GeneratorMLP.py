class GeneratorMLP(nn.Module):
    """Dense generator: a latent vector in, a 1 x 28 x 28 image out.

    Two choices are worth naming. The width grows layer after layer, 256 -> 512 ->
    1024 -> 784: the network starts from a very small description and has to invent
    detail, which is the opposite of a classifier. And the last activation is a
    `tanh`, with values in [-1, 1], because the images have been normalized to that
    range (convention 3). A sigmoid here, with images in [-1, 1], would make half of
    the target range unreachable.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

    def sample(self, n, device=device):
        return self.forward(torch.randn(n, self.latent_dim, device=device))
