vect_latent = [5, 10, 20, 30, 40, 50]
vect_train_loss = []
vect_val_loss = []

for n_latent in vect_latent:
    print("Latent space dimension: "+str(n_latent))
    autoencoder = Autoencoder_improved(n_input, n_latent)
    _, train_loss, val_loss = train_autoencoder(autoencoder, use_tqdm=False)
    vect_train_loss.append(train_loss)
    vect_val_loss.append(val_loss)
    print(" ")
    
# --- #

plt.plot(vect_latent, vect_train_loss, 'b', linestyle="--",label='Training loss')
plt.plot(vect_latent, vect_val_loss, 'g', label='Validation loss')

plt.title('Reconstruction loss')
plt.xlabel('epochs')
plt.legend()
plt.show()