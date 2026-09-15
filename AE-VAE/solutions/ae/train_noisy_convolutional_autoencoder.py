def train_noisy_convolutional_autoencoder(autoencoder,
                                          train_loader = train_noisy_loader,
                                          test_loader = test_noisy_loader,
                                          criterion = nn.BCELoss(),
                                          EPOCHS = 10,  #50
                                          LEARNING_RATE = 1e-3,
                                          device = device,
                                          use_tqdm = True
                                         ):
    """Train the autoencoder to go from the noisy image to the clean one.

    Two things change with respect to `train_convolutional_autoencoder`: the loaders
    return the triple (noisy, clean, label), and the target of the loss is the clean
    image, not the input. Everything the network learns comes from that asymmetry.
    """

    autoencoder = check_device_model(autoencoder, device=device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

    # --- #
    # Training loop
    for epoch in range(EPOCHS):
        autoencoder.train()
        train_loss = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False) if use_tqdm else train_loader
        for noisy_inputs, inputs, _ in iterator:
            noisy_inputs = noisy_inputs.to(device)
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = autoencoder(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy_inputs, inputs, _ in test_loader:
                noisy_inputs = noisy_inputs.to(device)
                inputs = inputs.to(device)
                outputs = autoencoder(noisy_inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return autoencoder, train_loss, val_loss
