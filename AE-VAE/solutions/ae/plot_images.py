def plot_images(imgs, sz, titles, n, cmap="gray"):
    """Display several rows of images, one row per entry of `imgs`.

    imgs   : list of batches, each a tensor or an array of at least `n` images
    sz     : shape each row must be reshaped to, e.g. (28, 28)
    titles : title of each row, placed above its first image
    n      : number of images displayed per row
    """
    num_rows = len(imgs)
    plt.figure(figsize=(2*n, 2*num_rows))

    for row, images in enumerate(imgs):
        # Accept tensors as well as arrays: bring everything back to numpy on the CPU
        if torch.is_tensor(images):
            images = images.detach().cpu().numpy()

        for i in range(n):
            ax = plt.subplot(num_rows, n, row*n + i + 1)
            plt.imshow(images[i].reshape(sz[row]), cmap=cmap)
            ax.axis("off")

        plt.subplot(num_rows, n, row*n + 1).set_title(titles[row], fontsize=12)

    plt.tight_layout()
    plt.show()
