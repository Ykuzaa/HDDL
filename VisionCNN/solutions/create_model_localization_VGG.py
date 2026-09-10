class LocalizationNetVGG(nn.Module):

    def __init__(self, conv_base, image_size=IMAGE_SIZE):
        super().__init__()
        self.conv_base = conv_base

        # Statistics of ImageNet, stored as buffers: they follow the model on the GPU,
        # but they are constants and not parameters to be learnt
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # VGG-16 has five pooling layers and ends with 512 channels
        n_features = 512 * (image_size // 32) ** 2

        self.head_p = nn.Linear(n_features, 1)
        self.head_coord = nn.Linear(n_features, 4)
        self.head_classes = nn.Linear(n_features, 4)

        # Only the heads are initialized: the convolutional base keeps its ImageNet weights
        self.head_p.apply(init_he_normal)
        self.head_coord.apply(init_he_normal)
        self.head_classes.apply(init_he_normal)

    def forward(self, x):
        x = (x - self.mean) / self.std
        f = torch.flatten(self.conv_base(x), 1)
        return self.head_p(f), self.head_coord(f), self.head_classes(f)