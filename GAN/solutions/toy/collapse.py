results = {}

# --- 1. the generator cannot keep up: twenty times the discriminator's step size
torch.manual_seed(SEED)
generator_slow = Generator(latent_dim=LATENT_DIM_2D, data_dim=2, hidden=128,
                           activation=nn.LeakyReLU(0.2))
discriminator_slow = Discriminator(data_dim=2, hidden=128,
                                   activation=nn.LeakyReLU(0.2))
train_gan_toy(generator_slow, discriminator_slow, sample_ring,
              n_steps=3000, lr_d=1e-3, lr_g=5e-5, optimizer="adam")
results["slow generator"] = generator_slow.sample(5000, device=device).detach()

# --- 2. the discriminator is given five steps per generator step, and a larger one
torch.manual_seed(SEED)
generator_strong = Generator(latent_dim=LATENT_DIM_2D, data_dim=2, hidden=128,
                             activation=nn.LeakyReLU(0.2))
discriminator_strong = Discriminator(data_dim=2, hidden=128,
                                     activation=nn.LeakyReLU(0.2))
train_gan_toy(generator_strong, discriminator_strong, sample_ring,
              n_steps=2000, lr_d=4e-3, lr_g=1e-3, optimizer="adam", n_d_steps=5)
results["strong discriminator"] = generator_strong.sample(5000, device=device).detach()

# --- and the balanced run of the previous section, for comparison
results["balanced"] = fake_samples

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
for ax, (name, samples) in zip(axes, results.items()):
    covered, quality, _ = mode_coverage(samples)
    plot_ring(samples, ax=ax,
              title=f"{name}\n{covered}/{N_MODES} modes, {quality:.0%} near a mode")
plt.tight_layout()
plt.show()

for name, samples in results.items():
    covered, quality, counts = mode_coverage(samples)
    print(f"{name:22s} {covered}/{N_MODES} modes, {quality:.1%} near a mode, {counts}")
