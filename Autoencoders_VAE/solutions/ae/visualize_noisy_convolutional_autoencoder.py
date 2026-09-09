def visualize_noisy_convolutional_autoencoder(autoencoder, 
                          test_noisy_loader = test_noisy_loader, 
                          input_sz = (28,28), 
                          latent_sz=(14,28),
                          n = 10, 
                          device = device
                         ):
    
    # Move autoencoder to device only if needed
    current_device = next(autoencoder.parameters()).device
    if current_device != torch.device(device):
        autoencoder.to(device)
    autoencoder.eval()

    dataset = test_noisy_loader.dataset
    indices = torch.randperm(len(dataset))[:n]
    noisy_images_list = [dataset[i][0].unsqueeze(0) for i in indices]
    images_list = [dataset[i][1].unsqueeze(0) for i in indices]
    
    noisy_inputs = torch.cat(noisy_images_list, dim=0).to(device)
    inputs = torch.cat(images_list, dim=0).to(device)
    with torch.no_grad():
        encoded = autoencoder.encoder(noisy_inputs)
        decoded = autoencoder.decoder(encoded)

    # Move to CPU and convert to numpy for plotting
    noisy_inputs, inputs, encoded, decoded = (x.cpu().numpy() for x in [noisy_inputs, inputs, encoded, decoded])

    
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(inputs[i].reshape(input_sz), cmap='gray')
        ax.axis('off')

        # Encoded (reshape approximately square)
        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(noisy_inputs[i].reshape(input_sz), cmap='gray')
        ax.axis('off')

        # Decoded
        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(decoded[i].reshape(input_sz), cmap='gray')
        ax.axis('off')

    title = ['test data','noisy images','denoised images']
    for i in range(3) :  
        ax = plt.subplot(3,n,1+i*n)
        plt.title(title[i])

    plt.tight_layout()
    plt.show()