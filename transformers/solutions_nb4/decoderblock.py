class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.self_attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = FeedForward(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    @staticmethod
    def add_causal_mask(mask):
        # assuming attn scores of shape (batch_size, num_heads, seq_len, seq_len)
        # assuming mask of shape (batch_size, 1, 1, seq_len)
        causal_mask = torch.triu(torch.ones(1, 1, mask.size(-1), mask.size(-1)), diagonal=1).type(torch.int)  # shape (1, 1, seq_len, seq_len)
        causal_mask = causal_mask == 0
        # Ensure the mask is on the same device as the input
        causal_mask.to(device)
        return mask & causal_mask

    def forward(self, x, enc_hidden_state, src_mask, tgt_mask):
        # Call the static method using the class name
        tgt_mask = TransformerDecoderBlock.add_causal_mask(tgt_mask)
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = x + attn_output
        x = self.norm1(x)
        attn_output = self.cross_attention(x, enc_hidden_state, enc_hidden_state, src_mask)
        x = x + attn_output
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm3(x)
        return x