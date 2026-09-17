class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # Compute the per-head hidden dimension

        self.W_query = nn.Linear(hidden_dim, hidden_dim) # queries weight matrix
        self.W_key = nn.Linear(hidden_dim, hidden_dim) # keys weight matrix
        self.W_value = nn.Linear(hidden_dim, hidden_dim) # values weight matrix

        self.W_out = nn.Linear(hidden_dim, hidden_dim)  # output weight matrix

    def forward(self, x):
        batch_size, num_tokens, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim, f"hidden_dim must be {self.hidden_dim}"

        keys = self.W_key(x)  # Shape: (batch_size, num_tokens, hidden_dim)
        queries = self.W_query(x) # Shape: (batch_size, num_tokens, hidden_dim)
        values = self.W_value(x) # Shape: (batch_size, num_tokens, hidden_dim)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention)
        attn_scores = torch.einsum("bijk, bikl -> bijl", queries, keys.transpose(2, 3))
        attn_weights = torch.softmax(attn_scores / (self.head_dim**0.5), dim=-1)

        # Attention output
        attn_output = torch.einsum("bijk, bikl -> bijl", attn_weights, values)

        # Transpose back: (batch_size, num_heads, num_tokens, head_dim) -> (batch_size, num_tokens, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Concatenate heads: (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_tokens, hiddend_dim)
        attn_output = attn_output.reshape(batch_size, num_tokens, self.hidden_dim)

        # Compute output
        output = self.W_out(attn_output)

        return output