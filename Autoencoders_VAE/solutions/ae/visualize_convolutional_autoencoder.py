def visualize_convolutional_autoencoder(autoencoder, 
                          loader = test_loader, 
                          input_sz = (28,28), 
                          latent_sz=(14,28),
                          n = 10, 
                          device = device
                         ):
    
    autoencoder = check_device_model(autoencoder, device=device)
    autoencoder.eval()

    inputs = select_n_samples(laoder=loader, n=n, device=device)
    with torch.no_grad():
        encoded = autoencoder.encoder(inputs)
        decoded = autoencoder.decoder(encoded)

    plot_images(
        imgs = [inputs, encoded, decoded],
        sz = [input_sz, latent_sz, input_sz],
        titles = ['Original', 'Encoded', 'Decoded'],
        n = n
    )