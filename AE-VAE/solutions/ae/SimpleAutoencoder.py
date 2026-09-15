class SimpleAutoencoder(nn.Module):
    """One dense layer down to the latent space, one dense layer back to the image.

    The decoder ends with a sigmoid so that its output lives in [0, 1], like the
    pixels of the normalized images: this is what makes the binary cross-entropy
    a legitimate reconstruction loss here.
    """

    def __init__(self, n_input, n_latent):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_latent),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_input),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Instantiate model
simple_autoencoder = SimpleAutoencoder(n_input, n_latent)
summary(simple_autoencoder, input_size=(1, n_input))
