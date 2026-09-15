class DCGenerator(nn.Module):
    """Convolutional generator, following the DCGAN recipe.

    The shape of the computation is the mirror image of the classifier of the CNN
    lab. There, an image was reduced to a vector by convolutions that halve the
    resolution; here a vector is turned into an image by transposed convolutions that
    double it: 7 x 7 -> 14 x 14 -> 28 x 28.

    `nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1)` is the layer that
    doubles a resolution: with a kernel of 4 and a padding of 1 the output side is
    exactly twice the input side, and the kernel size being a multiple of the stride
    avoids the checkerboard artefacts a kernel of 3 would produce.

    Three details are part of the recipe and not decoration: batch normalization on
    every hidden layer, `ReLU` in the generator (`LeakyReLU` is for the
    discriminator), and no normalization on the output layer, which ends with a
    `tanh`.
    """

    def __init__(self, latent_dim=LATENT_DIM, features=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = features

        # A latent vector has no spatial extent: it is first projected onto a
        # 7 x 7 map with 2*features channels, which is what the convolutions grow.
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 2 * features * 7 * 7),
            nn.BatchNorm1d(2 * features * 7 * 7),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * features, features, 4, stride=2, padding=1),
            nn.BatchNorm2d(features), nn.ReLU(),                      # 14 x 14
            nn.ConvTranspose2d(features, 1, 4, stride=2, padding=1),
            nn.Tanh(),                                                # 28 x 28
        )

    def forward(self, z):
        h = self.project(z).view(-1, 2 * self.features, 7, 7)
        return self.net(h)

    def sample(self, n, device=device):
        return self.forward(torch.randn(n, self.latent_dim, device=device))
