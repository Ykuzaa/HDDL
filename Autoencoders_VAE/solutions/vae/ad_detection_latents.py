fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

plot_latents(vae, beta=1, loader=test_ad_anomaly_loader, ensure_tsne=False, ax=axes[0])
plot_latents(vae, beta=1, loader=test_ad_anomaly_loader, ensure_tsne=True, ax=axes[1])
plt.show()