results_df = pd.DataFrame(results).T
results_df = results_df[['train_accuracy', 'val_accuracy', 'test_accuracy',
                         'train_loss', 'val_loss', 'test_loss',
                         'learning_time', 'prediction_time']]
results_df = results_df.sort_values('test_accuracy', ascending=False)

display(results_df.style.format("{:.4f}", subset=results_df.columns[:6])
                        .format("{:.0f} s", subset=['learning_time', 'prediction_time'])
                        .background_gradient(cmap='Greens', subset=['test_accuracy']))

# --- #

acc = results_df[['train_accuracy', 'val_accuracy', 'test_accuracy']]

ax = acc.plot.barh(figsize=(9, 0.9*len(acc) + 2))
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("Accuracy")
ax.axvline(0.5, color='grey', linestyle=':', label='chance level')
ax.set_title("Accuracy per model and per dataset")
plt.tight_layout()
plt.show()

# --- #

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(results_df['learning_time'], results_df['test_accuracy'])
for name, row in results_df.iterrows():
    ax.annotate(name, (row['learning_time'], row['test_accuracy']),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("Training time (s)")
ax.set_ylabel("Test accuracy")
ax.set_title("Accuracy vs. training cost")
plt.tight_layout()
plt.show()
