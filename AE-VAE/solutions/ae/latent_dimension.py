vect_latent = [5, 10, 20, 30, 40, 50]
vect_train_loss = []
vect_val_loss = []

# The loop variable is deliberately not called `n_latent`: that global is used further
# down the notebook and must keep the value fixed at the top of this part.
for dim in vect_latent:
    print("Latent space dimension: " + str(dim))
    autoencoder = Autoencoder_improved(n_input, dim)
    _, train_loss, val_loss = train_autoencoder(autoencoder, use_tqdm=False)
    vect_train_loss.append(train_loss)
    vect_val_loss.append(val_loss)
    print(" ")

# --- #

plt.plot(vect_latent, vect_train_loss, 'b', linestyle="--", label='Training loss')
plt.plot(vect_latent, vect_val_loss, 'g', label='Validation loss')

plt.title('Reconstruction loss')
plt.xlabel('latent space dimension')
plt.ylabel('binary cross-entropy')
plt.legend()
plt.show()
