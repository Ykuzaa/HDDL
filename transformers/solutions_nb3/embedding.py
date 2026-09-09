class Embedding(nn.Module):
  def __init__(self, vocab_size, max_length, hidden_dim):
    super().__init__()
    # TODO: use the nn.Embedding layer to initialize the embedding and positional encoding below:
    self.embedding = nn.Embedding(vocab_size, hidden_dim)
    self.position_encoding = nn.Embedding(max_length, hidden_dim)

  def forward(self, x):
    _, seq_length = x.shape
    token_embeddings = self.embedding(x)
    pos_encodings = self.position_encoding(torch.arange(seq_length, device=x.device))
    return token_embeddings + pos_encodings