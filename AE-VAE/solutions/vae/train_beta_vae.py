def train_beta_vae(vae,
                   beta = 1,
                   train_loader = train_loader,
                   test_loader = test_loader,
                   EPOCHS = 10,  #30
                   LEARNING_RATE = 1e-3,
                   device = device,
                   use_tqdm = True
                  ):
    """Train a beta-VAE and return it along with the history of the two terms of the loss.

    The history stores the reconstruction term and the KL term separately, both per image
    and **without** the factor beta on the KL. This is what makes two runs with different
    values of beta comparable: the total loss is not, since beta enters its definition.

    `kl_per_dim` holds, for each epoch, the contribution of each latent coordinate to the
    KL, measured on the validation set. A coordinate whose contribution is zero is one the
    encoder writes nothing into.
    """

    vae = check_device_model(vae, device=device)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    history = {k: [] for k in ['loss', 'recon', 'kl', 'val_loss', 'val_recon', 'val_kl',
                               'kl_per_dim']}

    # --- #
    # Training loop
    for epoch in range(EPOCHS):
        vae.train()
        train_recon, train_kl = 0., 0.

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False) if use_tqdm else train_loader
        for x, _ in iterator:
            x = x.view(-1, input_dim).to(device)
            optimizer.zero_grad()
            x_hat, z_mean, z_log_var = vae(x)
            recon, kl = vae_loss_terms(x, x_hat, z_mean, z_log_var)
            loss = recon + beta * kl
            loss.backward()
            optimizer.step()
            train_recon += recon.item()
            train_kl += kl.item()
        n_train = len(train_loader.dataset)
        train_recon, train_kl = train_recon / n_train, train_kl / n_train

        # Validation
        vae.eval()
        val_recon, val_kl, kl_dims = 0., 0., None
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.view(-1, input_dim).to(device)
                x_hat, z_mean, z_log_var = vae(x)
                recon, kl = vae_loss_terms(x, x_hat, z_mean, z_log_var)
                val_recon += recon.item()
                val_kl += kl.item()

                # Same KL, but summed over the batch only: one value per latent coordinate
                kl_j = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()).sum(dim=0)
                kl_dims = kl_j if kl_dims is None else kl_dims + kl_j
        n_val = len(test_loader.dataset)
        val_recon, val_kl = val_recon / n_val, val_kl / n_val

        history['recon'].append(train_recon)
        history['kl'].append(train_kl)
        history['loss'].append(train_recon + beta * train_kl)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)
        history['val_loss'].append(val_recon + beta * val_kl)
        history['kl_per_dim'].append((kl_dims / n_val).cpu().numpy())

        print(f"Epoch {epoch+1}/{EPOCHS}"
              f" - loss: {history['loss'][-1]:.2f}"
              f" - recon: {train_recon:.2f} - KL: {train_kl:.2f}"
              f" | val loss: {history['val_loss'][-1]:.2f}"
              f" - val recon: {val_recon:.2f} - val KL: {val_kl:.2f}")

    return vae, history
