def check_encode_decode(autoencoder, 
                        loader = test_loader, 
                        input_sz = (28,28), 
                        n = 10,
                        device = device
                       ):
    
    autoencoder = check_device_model(autoencoder, device=device)
    autoencoder.eval()
    
    inputs = select_n_samples(laoder=loader, n=n, device=device)  # [n, 1, 28, 28]
    inputs_flat = inputs.view(inputs.size(0), -1)                 # flatten for encoder/decoder

    with torch.no_grad():
        # Pass through the whole autoencoder
        autencoded = simple_autoencoder(inputs_flat)
    
        # Encode then decode explicitly
        encoded = simple_autoencoder.encoder(inputs_flat)
        decoded_encoded = simple_autoencoder.decoder(encoded)

    plot_images(
        imgs = [inputs, autencoded, decoded_encoded],
        sz = [input_sz, input_sz, input_sz],
        titles = ['test data','auto-endoded images','encoded -> decoded images'],
        n = n
    )