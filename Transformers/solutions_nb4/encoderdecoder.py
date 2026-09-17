class TransformerSummarizer(nn.Module):
    def __init__(self,
               vocab_size,
               encoder_max_length,
               decoder_max_length,
               hidden_dim,
               num_heads):
        super().__init__()

        self.src_embedding = Embedding(vocab_size, encoder_max_length, hidden_dim)
        self.tgt_embedding = Embedding(vocab_size, decoder_max_length, hidden_dim)
        self.encoder = TransformerEncoderBlock(hidden_dim, num_heads)
        self.decoder = TransformerDecoderBlock(hidden_dim, num_heads)
        self.summarizer_head = nn.Linear(hidden_dim, vocab_size)


    def encode(self, x_src, src_mask):
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        x_src = self.src_embedding(x_src)
        x_src = self.encoder(x_src, src_mask)
        return x_src


    def decode(self, x_tgt, encoder_hidden_state, src_mask, tgt_mask):
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)
        x_tgt = self.tgt_embedding(x_tgt)
        x_tgt = self.decoder(x_tgt, encoder_hidden_state, src_mask, tgt_mask)
        return x_tgt


    def forward(self, x_src, x_tgt, src_mask, tgt_mask):
        encoder_hidden_state = self.encode(x_src, src_mask)
        x_tgt = self.decode(x_tgt, encoder_hidden_state, src_mask, tgt_mask)
        logits = self.summarizer_head(x_tgt)
        return logits