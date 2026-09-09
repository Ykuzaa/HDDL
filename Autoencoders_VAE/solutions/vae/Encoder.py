class Encoder(nn.Module):
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


encoder = Encoder(input_dim, intermediate_dim, latent_dim)
summary(encoder, (input_dim,))