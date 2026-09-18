def plot_latents(vae, beta,
                 latent_dim = latent_dim,
                 loader = test_loader,
                 ensure_tsne = False,
                 ax = None):
    """Scatter plot of the encoded data, coloured by label.

    The point plotted is `z_mean`, not a sample of z: we want the position the encoder
    assigns to an image, not one draw of the noise around it.

    When the latent space has more than two dimensions, or when `ensure_tsne` is set,
    the points are projected with t-SNE. Beware that t-SNE preserves neighbourhoods,
    not distances: the size of the clusters and the gaps between them mean nothing.
    """

    vae = check_device_model(vae, device=device)
    vae.eval()

    latents = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            _, z_mean, _ = vae.encoder(images)
            latents.append(z_mean.cpu())
            labels_list.append(labels)

    latents = torch.cat(latents).numpy()
    labels_list = torch.cat(labels_list).numpy()

    if latent_dim != 2 or ensure_tsne:
        # t-SNE projection
        tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30)
        latents_2d = tsne.fit_transform(latents)
        title = f"Latent space (t-SNE projection), beta = {beta}"
    else:
        latents_2d = latents
        title = f"Latent space, beta = {beta}"

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
                         c=labels_list, cmap="tab10", alpha=0.6, s=5)
    ax.set_title(title)

    return scatter
