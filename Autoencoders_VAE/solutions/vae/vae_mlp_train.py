# Init model and optimizer
vae = VAE_mlp(input_dim, intermediate_dim, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

# --- #
# Training loop
for epoch in range(EPOCHS):
    vae.train()
    train_loss = 0
    
    for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        x = x.view(-1, input_dim).to(device)
        optimizer.zero_grad()
        x_hat, z_mean, z_log_var = vae(x)
        loss = vae_loss(x, x_hat, z_mean, z_log_var)
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
            loss = vae_loss(x, x_hat, z_mean, z_log_var)
            val_loss += loss.item()
    val_loss /= len(test_loader.dataset)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")vae_mlp_train