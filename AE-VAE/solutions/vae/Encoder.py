class VAEEncoder(nn.Module):
    """Encoder of the VAE, before the reparametrization trick.

    One shared hidden layer, then two heads reading it in parallel: one for the mean
    of the latent distribution, one for the logarithm of its variance. The log is what
    lets the network output any real number while the variance stays positive.
    """

    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc_mean = nn.Linear(intermediate_dim, latent_dim)
        self.fc_log_var = nn.Linear(intermediate_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var


vae_encoder = VAEEncoder(input_dim, intermediate_dim, latent_dim)
summary(vae_encoder, input_size=(1, input_dim))
