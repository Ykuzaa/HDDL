class CondGenerator(nn.Module):
    """Conditional generator: (noise, digit) in, an image of that digit out.

    The label is turned into a learned vector by `nn.Embedding` and concatenated to
    the latent vector. An embedding rather than the integer itself, because the digits
    have no order (7 is not between 6 and 8 in any sense the network should use), and
    rather than a one-hot vector, because an embedding *is* a one-hot vector followed
    by a linear layer, written efficiently.

    The rest is the DCGAN generator of Part II, with a latent dimension enlarged by
    `embed_dim`. That is the whole of the conditional GAN
    [Mirza & Osindero, 2014]: pass the label to both networks.
    """

    def __init__(self, latent_dim=LATENT_DIM, n_classes=10, embed_dim=32, features=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = features
        self.embedding = nn.Embedding(n_classes, embed_dim)

        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 2 * features * 7 * 7),
            nn.BatchNorm1d(2 * features * 7 * 7),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * features, features, 4, stride=2, padding=1),
            nn.BatchNorm2d(features), nn.ReLU(),
            nn.ConvTranspose2d(features, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        h = torch.cat([z, self.embedding(labels)], dim=1)
        h = self.project(h).view(-1, 2 * self.features, 7, 7)
        return self.net(h)

    def sample(self, labels, device=device):
        """One image per entry of `labels`."""
        labels = labels.to(device)
        z = torch.randn(labels.size(0), self.latent_dim, device=device)
        return self.forward(z, labels)
