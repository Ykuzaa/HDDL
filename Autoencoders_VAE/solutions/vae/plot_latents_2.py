LEARNING_RATE = 1e-3
EPOCHS = 20

betas = [.1, 1, 10]
fig, axes = plt.subplots(1, len(betas), figsize=(6*len(betas), 5), constrained_layout=True)

for i, beta in enumerate(betas):
    print(f"=== beta={beta} ===")
    vae = VAE_mlp(input_dim, intermediate_dim, latent_dim).to(device)
    vae, _, _ = train_beta_vae(vae, beta=beta, EPOCHS=EPOCHS, use_tqdm=False)
    scatter = plot_latents(vae, beta=beta, ax=axes[i])
    print('')

fig.colorbar(scatter, ax=axes, ticks=range(10))
plt.show()