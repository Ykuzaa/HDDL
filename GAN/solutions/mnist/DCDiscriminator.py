class DCDiscriminator(nn.Module):
    """Convolutional discriminator: a small classifier with a single output.

    Two departures from the CNN lab. There is no pooling: the resolution is halved by
    the stride of the convolutions themselves, 28 -> 14 -> 7, because pooling throws
    away exactly the local detail the discriminator is meant to judge. And the first
    layer carries no batch normalization, which is the DCGAN recipe again: normalizing
    the input layer would erase the contrast differences between a real batch and a
    fake one, which is part of what there is to detect.
    """

    def __init__(self, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),                                        # 14 x 14
            nn.Conv2d(features, 2 * features, 4, stride=2, padding=1),
            nn.BatchNorm2d(2 * features), nn.LeakyReLU(0.2),          # 7 x 7
        )
        self.head = nn.Linear(2 * features * 7 * 7, 1)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))
