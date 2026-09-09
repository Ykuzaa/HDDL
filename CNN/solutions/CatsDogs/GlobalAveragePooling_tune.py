# Final fine-tuning pass

epochs = 10

for p in conv_base.parameters():
    p.requires_grad = True

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam([                      # differentiated learning rates again
    {'params': vgg_combined_average[0].parameters(), 'lr': 1e-6},
    {'params': vgg_combined_average[2:].parameters(), 'lr': 1e-5},
])

t0 = time.time()
history_combined_average2 = fit(
    vgg_combined_average,
    train_loader_vgg_augmented,
    validation_loader_vgg,
    epochs=epochs,
    optimizer=optimizer,
    criterion=criterion,
    train_eval_loader=train_loader_vgg,   # honest train metric (see the note above)
    early_stopping_patience=5,
)
t_learning_gap += time.time() - t0

record("VGG + GlobalAvgPool (fine-tuned)", vgg_combined_average,
       (train_loader_vgg, validation_loader_vgg, test_loader_vgg),
       t_learning=t_learning_gap)
