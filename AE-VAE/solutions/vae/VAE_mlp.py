class VAE_mlp(nn.Module):
    """Variational autoencoder made of the encoder and the decoder written above.

    `forward` returns the three quantities the loss needs: the reconstruction, and the
    two parameters of q(z|x). The sampling of z happens inside the encoder.
    """

    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, intermediate_dim, latent_dim)
        self.decoder = VAEDecoder(input_dim, intermediate_dim, latent_dim)

    def forward(self, x):
        z, z_mean, z_log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z_mean, z_log_var


vae_mlp = VAE_mlp(input_dim, intermediate_dim, latent_dim)
summary(vae_mlp, input_size=(1, input_dim))
