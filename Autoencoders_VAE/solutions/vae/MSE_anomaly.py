def reconstruct(model, loader, device=device):
    model.eval()
    all_x = []
    all_x_reconstructed = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.view(x.size(0), -1).to(device)
            x_reconstructed, _, _ = model(x)

            all_x.append(x.cpu())
            all_x_reconstructed.append(x_reconstructed.cpu())

    all_x = torch.cat(all_x, dim=0).numpy()
    all_x_reconstructed = torch.cat(all_x_reconstructed, dim=0).numpy()
    return all_x, all_x_reconstructed


print("=== MSE over differents dataset ===")

# --- #
# Regular data
regular_data, regular_reconstructed = reconstruct(vae, test_ad_loader)
mse_regular = np.linalg.norm(regular_data-regular_reconstructed, axis=1)
print('Regular data:', np.mean(mse_normal))

# --- #
# Outliers data
outliers_data, outliers_reconstructed = reconstruct(vae, test_anomaly_loader)
mse_outliers = np.linalg.norm(outliers_data-outliers_reconstructed, axis=1)
print('Outliers:', np.mean(mse_outliers))

# --- #
# Random data

N = 1000
random_data = np.random.uniform(size=(N, 28, 28),low=0.0, high=1.0)
random_dataset = torch.tensor(random_data, dtype=torch.float32)
random_loader = DataLoader(
    random_dataset, 
    batch_size = BATCH_SIZE, 
    shuffle = False,
    num_workers = 2,
    pin_memory = True)

random_data, random_reconstructed = reconstruct(vae, random_loader)
mse_random = np.linalg.norm(random_data-random_reconstructed, axis=1)
print('Random data:', np.mean(mse_random))