def train_gan_toy(generator, discriminator, sample_real,
                  n_steps=3000, batch_size=256, lr=0.05, lr_g=None, lr_d=None,
                  optimizer="sgd", saturating=False, n_d_steps=1, record_every=25,
                  snapshot_steps=(), device=device):
    """Train a toy GAN, and return everything needed to look at the training afterwards.

    The loop is the one written by hand above, with three differences and no others:
    `nn.Module` holds the parameters, an `optim.Optimizer` performs the update, and
    `zero_grad()` clears the gradients. Read the body against `train_manual`: line
    for line, it is the same algorithm.

    `lr` sets both learning rates; `lr_g` and `lr_d` override it for one player. The
    two are separate on purpose: the balance between the players is a hyperparameter
    of its own, and section I.10 breaks the training by moving it.

    Returns
    -------
    history : dict of lists, one entry every `record_every` steps
    snapshots : dict, step -> generated samples at that step (detached, on CPU)
    """
    generator, discriminator = generator.to(device), discriminator.to(device)

    make = optim.Adam if optimizer == "adam" else optim.SGD
    kwargs = dict(betas=(0.5, 0.999)) if optimizer == "adam" else {}
    opt_g = make(generator.parameters(), lr=lr_g if lr_g is not None else lr, **kwargs)
    opt_d = make(discriminator.parameters(), lr=lr_d if lr_d is not None else lr, **kwargs)

    history = {k: [] for k in
               ["step", "loss_d", "loss_g", "grad_d", "grad_g", "d_real", "d_fake"]}
    snapshots = {}

    for step in range(n_steps):

        # ---- discriminator: tell the real batch from the fake one --------------
        for _ in range(n_d_steps):
            x_real = sample_real(batch_size).to(device)
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            x_fake = generator(z).detach()          # the generator is frozen here

            d_real = discriminator(x_real)
            d_fake = discriminator(x_fake)
            # -log D(x) - log(1 - D(G(z))), written with softplus
            loss_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

            opt_d.zero_grad()
            loss_d.backward()
            grad_d = grad_norm(discriminator.parameters())
            opt_d.step()

        # ---- generator: fool the discriminator as it is now --------------------
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        d_fake_g = discriminator(generator(z))      # no detach: the gradient must
                                                    # reach the generator
        loss_g = (-F.softplus(d_fake_g).mean() if saturating
                  else F.softplus(-d_fake_g).mean())

        opt_g.zero_grad()
        opt_d.zero_grad()            # the generator loss also fills the
        loss_g.backward()            # discriminator's gradients; they are cleared
        grad_g = grad_norm(generator.parameters())   # before the next D step
        opt_g.step()

        # ---- instruments -------------------------------------------------------
        if step % record_every == 0:
            history["step"].append(step)
            history["loss_d"].append(loss_d.item())
            history["loss_g"].append(loss_g.item())
            history["grad_d"].append(grad_d)
            history["grad_g"].append(grad_g)
            history["d_real"].append(torch.sigmoid(d_real).mean().item())
            history["d_fake"].append(torch.sigmoid(d_fake).mean().item())

        if step in snapshot_steps:
            with torch.no_grad():
                snapshots[step] = generator.sample(2000, device=device).cpu()

    return history, snapshots
