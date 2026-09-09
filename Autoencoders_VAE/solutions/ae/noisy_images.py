noisy_images, images, labels = next(iter(train_noisy_loader))
plot_images(
        imgs = [noisy_images, images],
        sz = [(28, 28), (28, 28)],
        titles = ['Noisy', 'Original'],
        n = n
    )