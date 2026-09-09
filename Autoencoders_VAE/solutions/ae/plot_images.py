def plot_images(imgs, sz, titles, n, cmap="gray"):
    num_rows = len(imgs)
    plt.figure(figsize=(2*n, 2*num_rows))

    for row, images in enumerate(imgs):
        # Move to CPU and convert to numpy for plotting
        images = images.cpu().numpy()
        
        for i in range(n):
            ax = plt.subplot(num_rows, n, row*n + i + 1)
            plt.imshow(images[i].reshape(sz[row]), cmap=cmap)
            ax.axis("off")
            
        plt.subplot(num_rows, n, row*n+1).set_title(titles[row], fontsize=12)

    plt.tight_layout()
    plt.show()