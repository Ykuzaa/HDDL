img_size = np.zeros((train_df.shape[0], 2))
for i, filename in enumerate(train_df['filename']):
    with Image.open(path + "train/train/" + filename) as img:
        img_size[i, :] = img.size          # PIL returns (width, height)

# --- #

plt.figure()
ax = sns.boxplot(img_size)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(['width', 'height'])
plt.title("Image width and height")
plt.show()
