from torch import nn
from torch.nn import Sequential


class MyTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim = 512, nhead = 8, num_encoder_layers = 6, num_decoder_layers = 6):
        super().__init__()


        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model = embed_dim,
            nhead=nhead,
            num_decoder_layers= num_decoder_layers,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=2048
        )

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        return self.transformer(src, tgt, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)