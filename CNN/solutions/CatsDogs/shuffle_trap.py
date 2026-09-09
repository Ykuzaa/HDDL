# A loader that shuffles — the obvious thing to do for a training set
train_loader_shuffled = make_loader(train_df, path + "train/train/",
                                    imagenet_norm=True, shuffle=True)

features_shuffled, y_from_loader = extract_features(conv_base, train_loader_shuffled)

# ... but labels read from the dataframe, in the order of the dataframe
y_from_df = torch.tensor(train_df['category'].astype(float).values, dtype=torch.float32)

print("Agreement between the two label vectors: %.3f"
      % (y_from_loader == y_from_df).float().mean().item())

# --- #

bad_loader = DataLoader(TensorDataset(features_shuffled, y_from_df),
                        batch_size=64, shuffle=True)
bad_eval_loader = DataLoader(TensorDataset(features_shuffled, y_from_df), batch_size=256)

vgg_mlp_bad = nn.Sequential(
    nn.Linear(train_features.shape[1], 256), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
).to(device)

optimizer = torch.optim.Adam(vgg_mlp_bad.parameters(), lr=3e-4)

history_bad = fit(
    vgg_mlp_bad, bad_loader, validation_features_loader,
    epochs=30, optimizer=optimizer,
    train_eval_loader=bad_eval_loader,
    restore_best_weights=False,
)

plot_training_analysis(history_bad)
