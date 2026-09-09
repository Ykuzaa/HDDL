class Decoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = F.relu(self.fc1(z))
        outputs = self.sigmoid (self.fc2(h))
        return outputs


decoder = Decoder(input_dim, intermediate_dim, latent_dim)
summary(decoder, (latent_dim,))