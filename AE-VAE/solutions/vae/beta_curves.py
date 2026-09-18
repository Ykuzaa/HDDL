fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(betas)))

for color, beta in zip(colors, betas):
    h = histories_by_beta[beta]
    epochs = range(1, len(h['recon']) + 1)
    axes[0].plot(epochs, h['recon'], color=color, label=f"beta = {beta}")
    axes[1].plot(epochs, h['kl'], color=color, label=f"beta = {beta}")

axes[0].set_title("Reconstruction term (BCE per image)")
axes[1].set_title("KL term per image, without the factor beta")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend()

plt.show()

# --- #
print(f"{'beta':>6} | {'recon':>9} | {'KL':>7} | {'total = recon + beta*KL':>24}")
for beta in betas:
    h = histories_by_beta[beta]
    print(f"{beta:>6} | {h['recon'][-1]:>9.2f} | {h['kl'][-1]:>7.3f} | {h['loss'][-1]:>24.2f}")
