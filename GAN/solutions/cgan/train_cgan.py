def train_cgan(generator, discriminator, loader, n_epochs=20, lr=2e-4,
               device=device, verbose=True):
    """The MNIST loop again, with the labels carried through.

    Three lines differ from `train_gan_mnist`, and they are all the same line: the
    label goes wherever the image goes. The real batch keeps its own labels; the fake
    batch is given labels drawn uniformly, so the generator is asked for every digit
    in roughly equal proportion.

    Drawing the fake labels uniformly rather than reusing the real ones is a choice.
    Reusing them would work, and would follow the class frequencies of MNIST, which
    are nearly uniform anyway; drawing them makes explicit that at generation time
    the label is an input we choose, not something that comes with the data.
    """
    generator, discriminator = generator.to(device), discriminator.to(device)
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Ten rows of ten, one digit per row, same noise throughout the training
    labels_fixed = torch.arange(10, device=device).repeat_interleave(10)
    z_fixed = torch.randn(100, generator.latent_dim, device=device)

    history = {k: [] for k in ["loss_d", "loss_g", "d_real", "d_fake"]}
    previews = []

    for epoch in range(n_epochs):
        running = np.zeros(4)
        n_batches = 0

        for x_real, y_real in tqdm(loader, desc=f"epoch {epoch + 1}/{n_epochs}",
                                   leave=False, disable=not verbose):
            x_real, y_real = x_real.to(device), y_real.to(device)
            batch_size = x_real.size(0)

            # ---- discriminator
            y_fake = torch.randint(0, 10, (batch_size,), device=device)
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            x_fake = generator(z, y_fake).detach()

            d_real = discriminator(x_real, y_real)
            d_fake = discriminator(x_fake, y_fake)
            loss_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---- generator
            y_fake = torch.randint(0, 10, (batch_size,), device=device)
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            d_fake_g = discriminator(generator(z, y_fake), y_fake)
            loss_g = F.softplus(-d_fake_g).mean()
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
            previews.append(generator(z_fixed, labels_fixed).cpu())
        generator.train()

        if verbose:
            print(f"epoch {epoch + 1:3d}/{n_epochs} | loss_d {running[0]:.3f} | "
                  f"loss_g {running[1]:.3f} | D(real) {running[2]:.2f} | "
                  f"D(fake) {running[3]:.2f}")

    return history, previews
