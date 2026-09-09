# %load solutions/vae/VAE_mlp.py

class VAE_mlp(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, intermediate_dim, latent_dim)
        self.decoder = Decoder(latent_dim, intermediate_dim, input_dim)

    def forward(self, x):
        z, z_mean, z_log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z_mean, z_log_var


vae_mlp = VAE_mlp(input_dim, intermediate_dim, latent_dim)
summary(vae_mlp, (input_dim,))