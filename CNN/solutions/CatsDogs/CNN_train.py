epochs = 30

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(cnn_simple.parameters(), lr=3e-4)

# Halve the learning rate after 3 epochs without progress on the validation loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

t_learning_cnn_simple_augmented = time.time()
cnn_simple_augmented_history = fit(
    cnn_simple,
    train_loader_augmented,
    validation_loader,
    epochs=epochs,
    optimizer=optimizer,
    criterion=criterion,
    train_eval_loader=train_loader,          # clean train metric, in eval() mode
    scheduler=scheduler,                     # ReduceLROnPlateau
    early_stopping_patience=8,               # EarlyStopping
    checkpoint_path="cnn_simple_best.pt",    # ModelCheckpoint
)
t_learning_cnn_simple_augmented = time.time() - t_learning_cnn_simple_augmented

print("Learning time for %d epochs : %d seconds" % (epochs, t_learning_cnn_simple_augmented))
