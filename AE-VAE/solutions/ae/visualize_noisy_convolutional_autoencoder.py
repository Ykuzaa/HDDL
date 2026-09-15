def visualize_noisy_convolutional_autoencoder(autoencoder,
                                              loader = test_noisy_loader,
                                              input_sz = (28, 28),
                                              n = 10,
                                              device = device
                                             ):
    """Display n test images, their noisy version, and what the autoencoder makes of it.

    `loader` must iterate over a NoisyMNIST dataset, whose items are the triple
    (noisy image, clean image, label): the clean image is what the first row shows.
    """

    autoencoder = check_device_model(autoencoder, device=device)
    autoencoder.eval()

    dataset = loader.dataset
    indices = torch.randperm(len(dataset))[:n]
    noisy_inputs = torch.cat([dataset[i][0].unsqueeze(0) for i in indices], dim=0).to(device)
    inputs = torch.cat([dataset[i][1].unsqueeze(0) for i in indices], dim=0).to(device)

    with torch.no_grad():
        decoded = autoencoder(noisy_inputs)

    plot_images(
        imgs = [inputs, noisy_inputs, decoded],
        sz = [input_sz, input_sz, input_sz],
        titles = ['Original', 'Noisy', 'Denoised'],
        n = n
    )
