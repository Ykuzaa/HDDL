def plot_latents(vae, beta,
                 latent_dim = latent_dim,
                 loader = test_loader,
                 ensure_tsne = False,
                 ax = None):

    vae.eval()
    latents = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1)
            _, z_mean, _ = vae.encoder(images)
            latents.append(z_mean)
            labels_list.append(labels)
    
    latents = torch.cat(latents).cpu().numpy()
    labels_list = torch.cat(labels_list).cpu().numpy()

    if latent_dim != 2 or ensure_tsne:
        # t-SNE projection
        tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30)
        latents_2d = tsne.fit_transform(latents)
        title = f"Latent space (t-SNE projection) -- beta = {beta}"
    else:
        latents_2d = latents
        title = f"Latent space -- beta = {beta}"

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels_list, cmap="tab10", alpha=0.6, s=5)
    ax.set_title(title)
    
    return scatter