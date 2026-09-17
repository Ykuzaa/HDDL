class TransformerEncoderBlock(nn.Module):
  def __init__(self, hidden_dim, num_heads):
    super(TransformerEncoderBlock, self).__init__()

    self.attention = MultiHeadAttention(hidden_dim, num_heads)
    self.norm1 = nn.LayerNorm(hidden_dim)
    self.norm2 = nn.LayerNorm(hidden_dim)
    self.feed_forward = FeedForward(hidden_dim)

  def forward(self, x):
      attn_output = self.attention(x)
      x = x + attn_output
      x = self.norm1(x)
      ff_output = self.feed_forward(x)
      x = x + ff_output
      x = self.norm2(x)
      return x