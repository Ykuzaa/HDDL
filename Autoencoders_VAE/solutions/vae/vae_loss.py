def vae_loss(x, x_hat, mean, log_var):
    reconstruct_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruct_loss + kl