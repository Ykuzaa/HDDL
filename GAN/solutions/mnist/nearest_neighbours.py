def nearest_neighbours(images, dataset, n_reference=10000, device=device):
    """For each generated image, the closest training image in the L2 pixel distance.

    The reference set is a subset of the training images, taken in order: with sixty
    thousand images of 784 pixels the full distance matrix is a few hundred megabytes,
    which is affordable, but the answer does not change and the wait does.
    """
    reference = torch.stack([dataset[i][0] for i in range(n_reference)])
    reference_flat = reference.view(n_reference, -1).to(device)
    images_flat = images.view(images.size(0), -1).to(device)

    distance = torch.cdist(images_flat, reference_flat)
    nearest, index = distance.min(dim=1)
    return reference[index.cpu()], nearest.cpu()


generated = generator_dc.sample(8, device=device).detach().cpu()
neighbours, distances = nearest_neighbours(generated, train_dataset)

plt.figure(figsize=(16, 4))
for i in range(8):
    ax = plt.subplot(2, 8, i + 1)
    plt.imshow(to_display(generated[i]), cmap="gray", vmin=0, vmax=1)
    ax.grid(False); plt.axis("off")
    if i == 0:
        ax.set_title("generated", loc="left", fontsize=10)

    ax = plt.subplot(2, 8, 8 + i + 1)
    plt.imshow(to_display(neighbours[i]), cmap="gray", vmin=0, vmax=1)
    ax.grid(False); plt.axis("off")
    plt.xlabel(f"{distances[i]:.1f}")
    if i == 0:
        ax.set_title("nearest training image", loc="left", fontsize=10)
plt.show()

print(f"L2 distance to the nearest training image: "
      f"{distances.min():.2f} to {distances.max():.2f}")
