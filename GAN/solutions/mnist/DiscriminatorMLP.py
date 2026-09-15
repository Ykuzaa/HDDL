class DiscriminatorMLP(nn.Module):
    """Dense discriminator: an image in, one logit out.

    `LeakyReLU` rather than `ReLU`, and dropout on both hidden layers. Both are there
    for the same reason: this network is not trying to win, it is trying to keep
    providing a usable gradient. A `ReLU` that switches off returns exactly zero to
    the generator, and a discriminator that becomes too good drives D(G(z)) to zero,
    where even the non-saturating loss has little left to say.
    """

    def __init__(self, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(p_drop),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(p_drop),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
