def vae_scores(model, loader, device=device):
    """Three per-image anomaly scores, computed in a single pass.

    - "MSE"   : mean square error between the image and its reconstruction;
    - "KL"    : Kullback-Leibler term alone, i.e. how far the code of this image sits
                from where the prior expects codes to be;
    - "-ELBO" : the sum of the two, with the reconstruction term as the binary
                cross-entropy summed over the pixels. This is the same quantity as the
                training loss, kept per image instead of summed over the batch.

    All three are oriented the same way: the larger, the more anomalous.
    """
    model = check_device_model(model, device=device)
    model.eval()

    mse, bce, kl = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device)
            x_hat, mean, log_var = model(x)

            mse.append((((x - x_hat) ** 2).mean(dim=1)).cpu())
            bce.append(F.binary_cross_entropy(x_hat, x, reduction="none").sum(dim=1).cpu())
            kl.append((-0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1)).cpu())

    mse = torch.cat(mse).numpy()
    bce = torch.cat(bce).numpy()
    kl = torch.cat(kl).numpy()

    return {"MSE": mse, "KL": kl, "-ELBO": bce + kl}


# --- #
scores_regular  = vae_scores(vae, test_ad_loader)
scores_outliers = vae_scores(vae, test_anomaly_loader)

true_label = np.concatenate([np.zeros(len(scores_regular["MSE"])),
                             np.ones(len(scores_outliers["MSE"]))])

fig, ax = plt.subplots(1, 1, figsize=(7, 6))

for name in ["MSE", "KL", "-ELBO"]:
    score = np.concatenate([scores_regular[name], scores_outliers[name]])
    fpr, tpr, _ = roc_curve(true_label, score)
    ax.plot(fpr, tpr, label=f"{name}, AUC = {roc_auc_score(true_label, score):.3f}")

ax.plot([0, 1], [0, 1], 'k--', label="Random guessing")
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title("Three scores, same test set")
ax.legend()
plt.show()
