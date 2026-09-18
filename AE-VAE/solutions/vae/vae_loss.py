def vae_loss(x, x_hat, mean, log_var):
    """Opposite of the ELBO, summed over the batch.

    `reduction="sum"` and not the default mean: the KL term below is a sum over the
    latent dimensions and over the batch, so the reconstruction term must be summed
    the same way for the two to be comparable. The division by the number of images
    is done once, in the training loop.
    """
    reconstruct_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruct_loss + kl
