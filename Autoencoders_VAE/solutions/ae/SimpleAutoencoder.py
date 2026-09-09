class SimpleAutoencoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super(SimpleAutoencoder, self).__init__()
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
summary(simple_autoencoder, input_size=(n_input,))