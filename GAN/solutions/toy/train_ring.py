torch.manual_seed(SEED)
generator_2d = Generator(latent_dim=LATENT_DIM_2D, data_dim=2, hidden=128,
                         activation=nn.LeakyReLU(0.2))
discriminator_2d = Discriminator(data_dim=2, hidden=128, activation=nn.LeakyReLU(0.2))

# Two changes with respect to the one-dimensional section, both deliberate. The hidden
# width goes from 64 to 128, because eight modes in two dimensions is a harder map to
# build than two modes in one. And the activation becomes LeakyReLU, the usual choice
# in a GAN: the tanh was there for the plain gradient descent of the previous section,
# and Adam does not need it.
history_2d, snapshots_2d = train_gan_toy(
    generator_2d, discriminator_2d, sample_ring,
    n_steps=5000, lr=1e-3, optimizer="adam",
    snapshot_steps=(0, 300, 1500, 4999))

fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
for ax, step in zip(axes, sorted(snapshots_2d)):
    plot_ring(snapshots_2d[step], ax=ax, title=f"step {step}")
plt.tight_layout()
plt.show()

fake_samples = generator_2d.sample(5000, device=device).detach()
plot_ring(fake_samples, discriminator_2d, title="after training, with $D$ behind")
plt.show()
