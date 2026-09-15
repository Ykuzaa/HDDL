class CondDiscriminator(nn.Module):
    """Conditional discriminator: (image, digit) in, one logit out.

    It now answers a harder question, "is this a real image *of this digit*?", and
    that is where the conditioning does its work. An image of a perfect 3 presented
    with the label 8 must be rejected, so the generator can no longer satisfy the
    discriminator by producing one convincing digit over and over.

    The label is turned into a whole extra channel, of constant value, and stacked
    onto the image. Concatenating along the channel axis lets the very first
    convolution combine label and pixels; giving the label to the final linear layer
    only would let the convolutions ignore it.
    """

    def __init__(self, n_classes=10, features=64):
        super().__init__()
        self.n_classes = n_classes
        self.embedding = nn.Embedding(n_classes, 28 * 28)

        self.net = nn.Sequential(
            nn.Conv2d(2, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),                                        # 14 x 14
            nn.Conv2d(features, 2 * features, 4, stride=2, padding=1),
            nn.BatchNorm2d(2 * features), nn.LeakyReLU(0.2),          # 7 x 7
        )
        self.head = nn.Linear(2 * features * 7 * 7, 1)

    def forward(self, x, labels):
        label_map = self.embedding(labels).view(-1, 1, 28, 28)
        h = torch.cat([x, label_map], dim=1)
        return self.head(self.net(h).flatten(1))
