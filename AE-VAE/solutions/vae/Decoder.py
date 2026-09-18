class VAEDecoder(nn.Module):
    """Decoder of the VAE: from the latent vector back to an image of `input_dim` pixels.

    The sigmoid on the output layer plays the same role as in the autoencoder lab: it brings
    the output back into [0, 1], which is what makes the binary cross-entropy legitimate.
    """

    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = F.relu(self.fc1(z))
        outputs = self.sigmoid(self.fc2(h))
        return outputs


vae_decoder = VAEDecoder(input_dim, intermediate_dim, latent_dim)
summary(vae_decoder, input_size=(1, latent_dim))
