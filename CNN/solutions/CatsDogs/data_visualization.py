x_batch, y_batch = next(iter(train_loader))

plt.figure(figsize=(12, 12))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_batch[i].permute(1, 2, 0))     # (C,H,W) -> (H,W,C)
    plt.title(labels.get(int(y_batch[i])))
    plt.axis('off')
    plt.grid(False)

plt.tight_layout()
plt.show()
