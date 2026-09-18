n_samples = 40

vae.eval()
with torch.no_grad():
    z = torch.randn(n_samples, latent_dim, device=device)
    generated = vae.decoder(z)

plot_images(
    imgs = [generated[k*10:(k+1)*10] for k in range(n_samples // 10)],
    sz = [(28, 28)] * (n_samples // 10),
    titles = [''] * (n_samples // 10),
    n = 10
)
