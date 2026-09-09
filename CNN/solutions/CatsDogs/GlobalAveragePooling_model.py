# Model definition

conv_base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device)

vgg_combined_average = nn.Sequential(
    conv_base,
    nn.AdaptiveAvgPool2d(1),   # (B, 512, 4, 4) -> (B, 512, 1, 1)
    nn.Flatten(),              # -> (B, 512)
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 1),
).to(device)

summary(vgg_combined_average, input_size=(1, 3, img_height, img_width))
