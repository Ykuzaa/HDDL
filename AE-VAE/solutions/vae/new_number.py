# The sweep over beta above left `vae` set to the last model it trained, the one with
# beta = 10. We retrain a beta = 1 VAE here, which is the one we want to generate from.
EPOCHS = 30
vae = VAE_mlp(input_dim, intermediate_dim, latent_dim).to(device)
vae, _ = train_beta_vae(vae, beta=1, EPOCHS=EPOCHS, use_tqdm=False)

# --- #
# A draw of the prior, on the same device as the model
z_latent = torch.randn(1, latent_dim, device=device)

vae.eval()
with torch.no_grad():
    x_generate = vae.decoder(z_latent)

x_generate = x_generate.cpu().numpy()
plt.imshow(x_generate[0].reshape(28, 28), cmap="gray")
plt.axis("off")
plt.show()
