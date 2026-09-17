class TransformerClassifier(nn.Module):
  def __init__(self,
               vocab_size,
               max_length,
               hidden_dim,
               num_heads,
               num_classes):
    super().__init__()

    self.embedding = Embedding(vocab_size, max_length, hidden_dim)
    self.encoder = TransformerEncoderBlock(hidden_dim, num_heads)
    self.classifier_head = nn.Linear(hidden_dim, num_classes)

  def forward(self, x):
    x = self.embedding(x)
    x = self.encoder(x)
    x = x[:, 0, :]
    x = self.classifier_head(x)
    return x