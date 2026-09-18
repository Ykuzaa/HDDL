def mse_per_image(x, x_reconstructed):
    """Mean square error between each image and its reconstruction, one value per image."""
    return ((x - x_reconstructed) ** 2).mean(axis=1)


print("=== Mean square error over the different datasets ===")

# --- #
# Regular data
regular_data, regular_reconstructed = reconstruct(vae, test_ad_loader)
mse_regular = mse_per_image(regular_data, regular_reconstructed)
print('Regular data:', np.mean(mse_regular))

# --- #
# Outliers data
outliers_data, outliers_reconstructed = reconstruct(vae, test_anomaly_loader)
mse_outliers = mse_per_image(outliers_data, outliers_reconstructed)
print('Outliers:', np.mean(mse_outliers))

# --- #
# Random data
N = 1000
random_dataset = torch.rand(N, 1, 28, 28)
random_loader = DataLoader(
    random_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 0,
    pin_memory = PIN_MEMORY)

random_data, random_reconstructed = reconstruct(vae, random_loader)
mse_random = mse_per_image(random_data, random_reconstructed)
print('Random data:', np.mean(mse_random))
