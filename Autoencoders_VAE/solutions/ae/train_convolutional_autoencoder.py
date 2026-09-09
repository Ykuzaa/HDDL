def train_convolutional_autoencoder(autoencoder,
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
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
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
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    return autoencoder, train_loss, val_loss