class LocalizationNet(nn.Module):

    def __init__(self, image_size=IMAGE_SIZE):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
        )

        # Four MaxPool2d(2): the spatial size is divided by 16, with 256 channels
        n_features = 256 * (image_size // 16) ** 2

        self.head_p = nn.Linear(n_features, 1)       # Output characterizing the presence of an object
        self.head_coord = nn.Linear(n_features, 4)   # Output characterizing bounding box coordinates
        self.head_classes = nn.Linear(n_features, 4) # Output characterizing the class probabilities

        self.apply(init_he_normal)

    def forward(self, x):
        # Flatten everything but the batch axis: (N, C, H, W) -> (N, C*H*W)
        f = torch.flatten(self.features(x), 1)
        return self.head_p(f), self.head_coord(f), self.head_classes(f)