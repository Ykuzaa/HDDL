class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128 * 2 * 2, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        return self.fc(z)