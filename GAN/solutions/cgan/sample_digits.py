def sample_digits(generator, digit, n=10, device=device):
    """Ten drawings of the same digit, from ten different latent vectors.

    This is the check the unconditional GAN of Part II could not pass: the label is an
    input, so the request "give me ten sevens" is now a single forward pass, and the
    ten images should differ from one another: same digit, different handwriting.
    """
    labels = torch.full((n,), digit, dtype=torch.long, device=device)
    generator.eval()
    with torch.no_grad():
        images = generator.sample(labels, device=device).cpu()
    generator.train()
    return images


plt.figure(figsize=(20, 2))
images = sample_digits(generator_c, digit=7, n=10)
for i, image in enumerate(images):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(to_display(image), cmap="gray", vmin=0, vmax=1)
    ax.grid(False)
    plt.axis("off")
plt.suptitle("Ten sevens, one latent vector each", y=1.05)
plt.show()

# Diversity, in one number: the mean pairwise distance inside the batch. A generator
# that has collapsed on a single seven would return something close to zero.
flat = images.view(images.size(0), -1)
print(f"mean pairwise L2 distance within the ten images: "
      f"{torch.cdist(flat, flat).sum() / (len(flat) * (len(flat) - 1)):.2f}")
