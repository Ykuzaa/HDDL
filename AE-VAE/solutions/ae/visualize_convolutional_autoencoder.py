def visualize_convolutional_autoencoder(autoencoder,
                                        loader = test_loader,
                                        input_sz = (28, 28),
                                        latent_sz = (14, 28),
                                        n = 10,
                                        device = device
                                       ):
    """Display n test images, their latent representation and their reconstruction.

    The latent representation of the convolutional autoencoder is a [8, 7, 7] volume,
    i.e. 392 values, laid out here as a 14x28 image so that it can be shown at all.
    """

    autoencoder = check_device_model(autoencoder, device=device)
    autoencoder.eval()

    inputs = select_n_samples(loader=loader, n=n, device=device)
    with torch.no_grad():
        encoded = autoencoder.encoder(inputs)
        decoded = autoencoder.decoder(encoded)

    plot_images(
        imgs = [inputs, encoded, decoded],
        sz = [input_sz, latent_sz, input_sz],
        titles = ['Original', 'Encoded', 'Decoded'],
        n = n
    )
