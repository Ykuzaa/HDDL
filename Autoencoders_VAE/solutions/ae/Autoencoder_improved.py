# Decoder
class Decoder_improved(nn.Module):
    def __init__(self, n_latent, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_latent, 128)
        self.fc2 = nn.Linear(128, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Autoencoder
class Autoencoder_improved(nn.Module):
    def __init__(self, n_input, n_latent):
        super().__init__()
        self.encoder = Encoder(n_input, n_latent)
        self.decoder = Decoder_improved(n_latent, n_input)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# --- #


def train_autoencoder(autoencoder,
                      train_loader = train_loader,
                      test_loader = test_loader,
                      criterion = nn.BCELoss(),
                      EPOCHS = 10,  #50
                      LEARNING_RATE = 1e-3,
                      device = device,
                      use_tqdm = True
                     ):

    autoencoder = check_device_model(autoencoder, device=device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    
    # --- #
    # Training loop
    for epoch in range(EPOCHS):
        autoencoder.train()
        train_loss = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False) if use_tqdm else train_loader
        for inputs, _ in iterator:
            inputs = inputs.to(device)
            inputs_flat = inputs.view(inputs.size(0), -1)
            
            optimizer.zero_grad()
            outputs = autoencoder(inputs_flat)
            loss = criterion(outputs, inputs_flat)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                inputs_flat = inputs.view(inputs.size(0), -1)
                outputs = autoencoder(inputs_flat)
                loss = criterion(outputs, inputs_flat)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    return autoencoder, train_loss, val_loss
    

# --- #


autoencoder = Autoencoder_improved(n_input, n_latent)
autoencoder, _, _ = train_autoencoder(autoencoder)
visualize_autoencoder(autoencoder)

