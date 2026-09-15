noisy_images, images, labels = next(iter(train_noisy_loader))

plot_images(
    imgs = [images, noisy_images],
    sz = [(28, 28), (28, 28)],
    titles = ['Original', 'Noisy'],
    n = 10
)
