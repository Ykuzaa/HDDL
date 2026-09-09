vgg_mlp = nn.Sequential(
    nn.Linear(train_features.shape[1], 256), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
).to(device)

summary(vgg_mlp, input_size=(1, train_features.shape[1]))
