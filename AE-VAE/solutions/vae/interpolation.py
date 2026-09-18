def interpolate(model, x_a, x_b, n_steps=10, variational=False):
    """Decode along the segment joining the codes of two images.

    For the VAE we interpolate between the two **means** of q(z|x), not between two
    samples: we want the position the encoder assigns to each image, not one draw.
    """
    model = check_device_model(model, device=device)
    model.eval()

    with torch.no_grad():
        if variational:
            _, z_a, _ = model.encoder(x_a)
            _, z_b, _ = model.encoder(x_b)
        else:
            z_a = model.encoder(x_a)
            z_b = model.encoder(x_b)

        alphas = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)
        z = (1 - alphas) * z_a + alphas * z_b
        return model.decoder(z)


# --- #
decoded_ae  = interpolate(autoencoder_2d, x_a, x_b, n_steps=N_STEPS, variational=False)
decoded_vae = interpolate(vae,            x_a, x_b, n_steps=N_STEPS, variational=True)

plot_images(
    imgs = [decoded_ae, decoded_vae],
    sz = [(28, 28), (28, 28)],
    titles = ['Autoencoder', 'Variational autoencoder'],
    n = N_STEPS
)
