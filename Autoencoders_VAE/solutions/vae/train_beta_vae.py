def train_beta_vae(vae,
              beta = 1,
              train_loader = train_loader,
              test_loader = test_loader,
              EPOCHS = 10,  #30
              LEARNING_RATE = 1e-3,
              device = device,
              use_tqdm = True
             ):
    
    vae = check_device_model(vae, device=device)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    # --- #
    # Training loop
    for epoch in range(EPOCHS):
        vae.train()
        train_loss = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False) if use_tqdm else train_loader
        for x, _ in iterator:
            x = x.view(-1, input_dim).to(device)
            optimizer.zero_grad()
            x_hat, z_mean, z_log_var = vae(x)
            loss = beta_vae_loss(x, x_hat, z_mean, z_log_var, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
    
        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.view(-1, input_dim).to(device)
                x_hat, z_mean, z_log_var = vae(x)
                loss = beta_vae_loss(x, x_hat, z_mean, z_log_var, beta=beta)
                val_loss += loss.item()
        val_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return vae, train_loss, val_loss