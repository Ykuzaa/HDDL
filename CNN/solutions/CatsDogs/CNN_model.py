cnn_simple = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(64, 96, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    # nn.Conv2d(96, 128, kernel_size=3), nn.ReLU(),
    # nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    # nn.Linear(96*17*17, 512), nn.ReLU(),
    nn.Linear(96*17*17, 64), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1),          # no sigmoid: BCEWithLogitsLoss will take care of it
).to(device)

summary(cnn_simple, input_size=(1, 3, img_height, img_width))
