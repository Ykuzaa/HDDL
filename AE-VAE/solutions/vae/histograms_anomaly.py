# Histograms
fig = plt.figure(figsize=(9, 5))
ax = plt.subplot(1, 1, 1)

sns.histplot(data=mse_regular, stat='density', color="skyblue", ax=ax, label="Normal", kde=True)
sns.histplot(data=mse_outliers, stat='density', color="purple", ax=ax, label="Outliers", kde=True)
sns.histplot(data=mse_random, stat='density', color="teal", ax=ax, label="Random", kde=True)

ax.set_xlabel('Reconstruction error (MSE)')
plt.legend()
plt.show()
