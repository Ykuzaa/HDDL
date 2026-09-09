from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

test_prediction, y_test = predict(vgg_combined, test_loader_vgg)
y_pred = (test_prediction > 0.5).astype(int)

cm = confusion_matrix(y_test.astype(int), y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.show()

print(classification_report(y_test.astype(int), y_pred, target_names=['Cat', 'Dog']))
print("ROC AUC: %.4f" % roc_auc_score(y_test, test_prediction))

# --- Worst mistakes: wrong, and confident about it ---
confidence_in_truth = np.where(y_test == 1, test_prediction, 1 - test_prediction)
worst = np.argsort(confidence_in_truth)[:9]

fig = plt.figure(figsize=(10, 10))
for i, idx in enumerate(worst):
    img = Image.open(path + "test/" + test_df['filename'][idx]).convert("RGB")
    ax = fig.add_subplot(3, 3, i+1)
    ax.imshow(img, interpolation='nearest')
    ax.axis('off')
    ax.set_title("True: %s - P(dog) = %.3f"
                 % (labels[int(y_test[idx])], test_prediction[idx]), color="red")

plt.suptitle("The 9 most confident mistakes")
plt.tight_layout()
plt.show()
