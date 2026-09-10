class YOLONet(nn.Module):

    def __init__(self, image_size=IMAGE_SIZE):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding='same'), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding='same'), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding='same'), nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding='same'), nn.ELU(),
            nn.Conv2d(64, 64, 3, padding='same'), nn.ELU(),
            nn.Conv2d(64, 64, 3, padding='same'), nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding='same'), nn.ELU(),
            nn.Conv2d(128, 128, 3, padding='same'), nn.ELU(),
            nn.Conv2d(128, 128, 3, padding='same'), nn.ELU(),
            nn.MaxPool2d(2),
        )

        # Three MaxPool2d(2): the spatial size is divided by 8, with 128 channels
        n_features = 128 * (image_size // 8) ** 2

        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, CELL_PER_DIM * CELL_PER_DIM * (NB_CLASSES + 5*BOX_PER_CELL)),
        )

        self.apply(init_he_normal)

    def forward(self, x):
        f = torch.flatten(self.features(x), 1)
        output = self.classifier(f)
        return output.view(-1, CELL_PER_DIM, CELL_PER_DIM, NB_CLASSES + 5*BOX_PER_CELL)