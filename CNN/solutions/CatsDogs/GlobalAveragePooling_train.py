# Training the last dense layers of the network

epochs = 20

for p in conv_base.parameters():
    p.requires_grad = False

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    [p for p in vgg_combined_average.parameters() if p.requires_grad], lr=3e-4)

t0 = time.time()
history_combined_average = fit(
    vgg_combined_average,
    train_loader_vgg_augmented,
    validation_loader_vgg,
    epochs=epochs,
    optimizer=optimizer,
    criterion=criterion,
    train_eval_loader=train_loader_vgg,   # honest train metric (see the note above)
    early_stopping_patience=6,
    checkpoint_path="vgg_gap_best.pt",
)
t_learning_gap = time.time() - t0
