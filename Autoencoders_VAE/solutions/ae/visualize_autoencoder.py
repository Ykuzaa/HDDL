def visualize_autoencoder(autoencoder, 
                          loader = test_loader, 
                          input_sz = (28,28), 
                          latent_sz = (4,8),
                          n = 10, 
                          device = device
                         ):
    
    # Move autoencoder to device only if needed
    current_device = next(autoencoder.parameters()).device
    if current_device != torch.device(device):
        autoencoder.to(device)
    autoencoder.eval()

    dataset = loader.dataset
    indices = torch.randperm(len(dataset))[:n]
    images_list = [dataset[i][0].unsqueeze(0) for i in indices]
    
    inputs = torch.cat(images_list, dim=0).to(device)  # [n, 1, 28, 28]
    inputs_flat = inputs.view(inputs.size(0), -1)      # flatten for encoder/decoder


    with torch.no_grad():
        encoded = autoencoder.encoder(inputs_flat)
        decoded = autoencoder.decoder(encoded)

    # Move to CPU and convert to numpy for plotting
    inputs, encoded, decoded = (x.cpu().numpy() for x in [inputs, encoded, decoded])

    
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(inputs[i].reshape(input_sz), cmap='gray')
        ax.axis('off')

        # Encoded (reshape approximately square)
        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(encoded[i].reshape(latent_sz), cmap='gray')
        ax.axis('off')

        # Decoded
        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(decoded[i].reshape(input_sz), cmap='gray')
        ax.axis('off')

    title = ['test data','endoded images','decoded images']
    for i in range(3) :  
        ax = plt.subplot(3,n,1+i*n)
        plt.title(title[i])

    plt.tight_layout()
    plt.show()