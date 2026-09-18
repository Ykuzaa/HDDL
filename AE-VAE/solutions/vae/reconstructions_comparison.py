# Regular test data (digits 0 to 8)
print("=== Regular data ===")
test_ad_data, test_ad_reconstructed = reconstruct(vae, test_ad_loader)
plot_images(
    imgs = [test_ad_data, test_ad_reconstructed],
    sz = [(28, 28), (28, 28)],
    titles = ['Original', 'Decoded'],
    n = 10
)

# Outliers (the 9s, never seen during training)
print("=== Outliers ===")
test_anomaly_data, test_anomaly_reconstructed = reconstruct(vae, test_anomaly_loader)
plot_images(
    imgs = [test_anomaly_data, test_anomaly_reconstructed],
    sz = [(28, 28), (28, 28)],
    titles = ['Original', 'Decoded'],
    n = 10
)
