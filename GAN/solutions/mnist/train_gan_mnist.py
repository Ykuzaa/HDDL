def train_gan_mnist(generator, discriminator, loader, n_epochs=20, lr=2e-4,
                    saturating=False, device=device, n_preview=64, verbose=True):
    """The toy loop again, with a `DataLoader` in place of `sample_real`.

    Nothing in the algorithm changes when the data becomes images. What changes is
    that the real batches now come from a finite dataset, so the loop runs over
    epochs, and that the samples are worth looking at: a *fixed* latent batch is
    decoded at the end of every epoch, so the previews form a film of the same
    sixty-four images being learned, and not sixty-four unrelated draws.

    Adam with betas = (0.5, 0.999) is the DCGAN recipe
    [Radford, Metz & Chintala, ICLR 2016]. The default beta1 of 0.9 gives too much
    inertia to a loss whose landscape moves at every step.
    """
    generator, discriminator = generator.to(device), discriminator.to(device)
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    z_fixed = torch.randn(n_preview, generator.latent_dim, device=device)
    history = {k: [] for k in ["loss_d", "loss_g", "d_real", "d_fake"]}
    previews = []

    for epoch in range(n_epochs):
        running = np.zeros(4)
        n_batches = 0

        for x_real, _ in tqdm(loader, desc=f"epoch {epoch + 1}/{n_epochs}",
                              leave=False, disable=not verbose):
            x_real = x_real.to(device)
            batch_size = x_real.size(0)

            # ---- discriminator
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            x_fake = generator(z).detach()
            d_real, d_fake = discriminator(x_real), discriminator(x_fake)
            loss_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---- generator
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            d_fake_g = discriminator(generator(z))
            loss_g = (-F.softplus(d_fake_g).mean() if saturating
                      else F.softplus(-d_fake_g).mean())
            opt_g.zero_grad()
            opt_d.zero_grad()
            loss_g.backward()
            opt_g.step()

            running += [loss_d.item(), loss_g.item(),
                        torch.sigmoid(d_real).mean().item(),
                        torch.sigmoid(d_fake).mean().item()]
            n_batches += 1

        running /= n_batches
        for key, value in zip(history, running):
            history[key].append(value)

        generator.eval()
        with torch.no_grad():
            previews.append(generator(z_fixed).cpu())
        generator.train()

        if verbose:
            print(f"epoch {epoch + 1:3d}/{n_epochs} | loss_d {running[0]:.3f} | "
                  f"loss_g {running[1]:.3f} | D(real) {running[2]:.2f} | "
                  f"D(fake) {running[3]:.2f}")

    return history, previews
