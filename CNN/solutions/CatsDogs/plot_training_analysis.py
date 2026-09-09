def plot_training_analysis(history, figsize=(12, 4)):
    epochs = range(len(history['accuracy']))
    has_clean = 'clean_accuracy' in history

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'b', linestyle="--",
             label='Training accuracy (on the fly)')
    if has_clean:
        plt.plot(epochs, history['clean_accuracy'], 'r', linestyle=':',
                 label='Training accuracy (clean, eval mode)')
    plt.plot(epochs, history['val_accuracy'], 'g', label='Validation accuracy')
    plt.title('Model accuracy')
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'b', linestyle="--", label='Training loss (on the fly)')
    if has_clean:
        plt.plot(epochs, history['clean_loss'], 'r', linestyle=':',
                 label='Training loss (clean, eval mode)')
    plt.plot(epochs, history['val_loss'], 'g', label='Validation loss')
    plt.title('Model loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
