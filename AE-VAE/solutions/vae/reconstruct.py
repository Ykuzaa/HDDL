def reconstruct(model, loader, device=device):
    """Pass a whole loader through the VAE and return the originals and their reconstructions.

    Both come back as numpy arrays of shape [N, 784], in the order of the loader.
    `loader` may yield either plain tensors or (image, label) pairs.
    """
    model = check_device_model(model, device=device)
    model.eval()

    all_x = []
    all_x_reconstructed = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device)
            x_reconstructed, _, _ = model(x)

            all_x.append(x.cpu())
            all_x_reconstructed.append(x_reconstructed.cpu())

    all_x = torch.cat(all_x, dim=0).numpy()
    all_x_reconstructed = torch.cat(all_x_reconstructed, dim=0).numpy()
    return all_x, all_x_reconstructed
