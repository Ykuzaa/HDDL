train = True
if train:
    EPOCHS = 30
    vae = VAE_mlp(input_dim, intermediate_dim, latent_dim).to(device)
    vae, _, _ = train_beta_vae(vae, beta=1, EPOCHS=EPOCHS, use_tqdm=False)

z_latent = np.random.normal(0, 1, (1, latent_dim)).astype(np.float32)
z_latent = torch.from_numpy(z_latent)

vae.decoder.eval()
with torch.no_grad():
    x_generate = vae.decoder(z_latent)

x_generate = x_generate.cpu().numpy()
plt.imshow(x_generate[0].reshape(28, 28), cmap="gray")
plt.axis("off")
plt.show()