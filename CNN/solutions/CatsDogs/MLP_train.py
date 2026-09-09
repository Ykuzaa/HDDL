epochs = 50

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(vgg_mlp.parameters(), lr=3e-4)

t_learning_vgg_mlp = time.time()
vgg_mlp_history = fit(
    vgg_mlp,
    train_features_loader,
    validation_features_loader,
    epochs=epochs,
    optimizer=optimizer,
    criterion=criterion,
    train_eval_loader=train_features_eval_loader,
    early_stopping_patience=10,
    checkpoint_path="vgg_mlp_best.pt",
)
t_learning_vgg_mlp = time.time() - t_learning_vgg_mlp

print("Learning time for %d epochs : %d seconds" % (epochs, t_learning_vgg_mlp))
