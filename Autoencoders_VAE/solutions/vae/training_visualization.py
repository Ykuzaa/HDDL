# Init model and optimizer
vae_seq = VAE_mlp(input_dim, intermediate_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(vae_seq.parameters(), lr=LEARNING_RATE)

# --- #  
# Select test images for plotting
idx = torch.randint(0, len(test_dataset), (n,))

test_images = []
for i in idx:
    img, _ = test_dataset[i]
    test_images.append(img)
x_test_sample = torch.stack(test_images).view(n, -1).to(device)

imgs = [x_test_sample]
titles = ['Images']


# --- #  
# Reconstruction before training (epoch 0)
vae_seq.eval()
with torch.no_grad():
    x_test_decoded = vae_seq(x_test_sample)[0].view(n, 28, 28)

imgs.append(x_test_decoded)
titles.append('Init')


# --- #
# Training loop
for j in range(m):
    print(f"=== Epoch {j+1}/{m} ===", end="\r", flush=True)
    vae_seq.train()
    for x_batch, _ in train_loader:
        x_batch = x_batch.view(x_batch.size(0), -1).to(device)
        optimizer.zero_grad()
        x_hat, z_mean, z_log_var = vae_seq(x_batch)
        loss = vae_loss(x_batch, x_hat, z_mean, z_log_var)
        loss.backward()
        optimizer.step()

    # Reconstruction at epoch j
    vae_seq.eval()
    with torch.no_grad():
        x_test_decoded = vae_seq(x_test_sample)[0].view(n, 28, 28)

    imgs.append(x_test_decoded)
    titles.append(f'Epoch {j+1}')
    
print("Training complete!")

sz = [(28,28)] * len(imgs)
plot_images(imgs, sz, titles, n)