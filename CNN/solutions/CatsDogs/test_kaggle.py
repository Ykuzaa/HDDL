test_prediction, y_test = predict(vgg_combined, test_loader_vgg)
test_accuracy = ((test_prediction > 0.5).astype(float) == y_test).mean()
print('Test accuracy:', test_accuracy)

# --- #

fig = plt.figure(figsize=(10, 10))

test_imgs_idx = np.random.randint(low=0, high=test_df.shape[0], size=(9,))

for i, idx in enumerate(test_imgs_idx):
    img = Image.open(path + "test/" + test_df['filename'][idx]).convert("RGB")
    pred = test_prediction[idx]

    ax = fig.add_subplot(3, 3, i+1)
    ax.imshow(img, interpolation='nearest')
    ax.axis('off')
    color = "green"
    if pred > 0.5:
        title = "Probability for dog : %.1f" % (pred*100)
        if test_df['category'][idx] == '0':
            color = "red"
    else:
        title = "Probability for cat : %.1f" % ((1-pred)*100)
        if test_df['category'][idx] == '1':
            color = "red"
    ax.set_title(title, color=color)

plt.tight_layout()
plt.show()
