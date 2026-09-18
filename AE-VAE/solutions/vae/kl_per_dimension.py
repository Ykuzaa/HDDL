fig, ax = plt.subplots(figsize=(10, 5))

width = 0.8 / len(betas)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(betas)))

for k, (color, beta) in enumerate(zip(colors, betas)):
    # Sorted by decreasing contribution: the numbering of the coordinates means nothing,
    # only how many of them carry something does.
    kl_j = np.sort(histories_10d[beta]['kl_per_dim'][-1])[::-1]
    ax.bar(np.arange(len(kl_j)) + k*width, kl_j, width=width, color=color, label=f"beta = {beta}")

ax.set_xticks(np.arange(LATENT_DIM_FAR) + (len(betas) - 1)*width/2)
ax.set_xticklabels(range(1, LATENT_DIM_FAR + 1))
ax.set_xlabel("latent coordinates, sorted by decreasing contribution")
ax.set_ylabel("KL of that coordinate, in nats per image")
ax.legend()
plt.show()

# --- #
THRESHOLD = 0.01  # nats per image: below this, a coordinate carries nothing

print(f"{'beta':>6} | {'active coordinates':>19} | {'total KL':>9}")
for beta in betas:
    kl_j = np.asarray(histories_10d[beta]['kl_per_dim'][-1])
    print(f"{beta:>6} | {int((kl_j > THRESHOLD).sum()):>19} | {kl_j.sum():>9.3f}")
