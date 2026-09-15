class ConvEncoder(nn.Module):
    """Two convolutions, a max-pooling, two convolutions, a max-pooling: 28 -> 14 -> 7."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)  # 28 -> 14

        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)  # 14 -> 7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        return x


class ConvDecoder(nn.Module):
    """Mirror of the encoder, the max-poolings replaced by upsamplings: 7 -> 14 -> 28."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 8, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding='same')
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # 7 -> 14

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding='same')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')  # 14 -> 28

        self.conv5 = nn.Conv2d(16, 1, kernel_size=3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.up1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.up2(x)

        x = self.conv5(x)
        x = self.sigmoid(x)
        return x


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder. Its latent representation is a volume, [B, 8, 7, 7],
    and not a vector: this is what changes in the training and display functions."""

    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Instantiate model
convolutional_autoencoder = ConvAutoencoder()
summary(convolutional_autoencoder, input_size=(1, 1, 28, 28))
