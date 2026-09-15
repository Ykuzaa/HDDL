def interpolate(generator, z_start, z_end, n_steps=10, device=device):
    """Decode the straight segment between two latent vectors.

    The interpolation is linear in the latent space, which is the usual thing to do
    and is slightly wrong: with a standard Gaussian prior in dimension 100, almost all
    the mass sits in a thin shell of radius sqrt(100) = 10, and the midpoint of two
    such vectors has a norm of about 7. The middle of the segment therefore lands in a
    region the generator saw very little of during training. A spherical interpolation
    would stay on the shell; the difference is visible on faces, much less on MNIST.
    """
    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1)
    z = (1 - alphas) * z_start.view(1, -1) + alphas * z_end.view(1, -1)

    generator.eval()
    with torch.no_grad():
        images = generator(z).cpu()
    generator.train()
    return images


torch.manual_seed(SEED)
z_a, z_b = torch.randn(2, LATENT_DIM, device=device)
images = interpolate(generator_dc, z_a, z_b, n_steps=10)

plt.figure(figsize=(20, 2))
for i, image in enumerate(images):
    ax = plt.subplot(1, len(images), i + 1)
    plt.imshow(to_display(image), cmap="gray", vmin=0, vmax=1)
    ax.grid(False)
    plt.axis("off")
plt.suptitle("Linear interpolation in the latent space", y=1.05)
plt.show()
