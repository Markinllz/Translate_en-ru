from torch import nn
from torch.nn import Sequential
import torch


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
        
        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        src = self.src_embedding(src)
        
        if src_mask is not None:
            src_mask = src_mask.bool()
        if tgt_mask is not None:
            tgt_mask = tgt_mask.bool()
        
        if tgt is not None:
            tgt = self.tgt_embedding(tgt)
            output = self.transformer(src, tgt, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        else:
            output = self.transformer.encoder(src, src_key_padding_mask=src_mask)
        
        logits = self.output_projection(output)
        
        return type('Output', (), {'logits': logits})()
    

    def generate(self, src, src_mask=None, max_length=50):
        """
        Инференс для одного предложения.
        - src: (src_seq_len,)
        - src_mask: (src_seq_len,), если есть паддинг
        Возвращает: (generated_seq_len,)
        """
        
        if src.dim() == 1:
            src = src.unsqueeze(1)  
        
        if src_mask is not None and src_mask.dim() == 1:
            src_mask = src_mask.unsqueeze(0) 
        
        device = src.device
        memory = self.transformer.encoder(self.src_embedding(src), 
                                         src_key_padding_mask=src_mask)
        
        
        tgt = torch.tensor([[0]], dtype=torch.long, device=device)  
        
        for _ in range(max_length):
            output = self.transformer.decoder(
                self.tgt_embedding(tgt), 
                memory,
                memory_key_padding_mask=src_mask
            )
            logits = self.output_projection(output[-1:])  
            next_token = logits.argmax(-1)
            tgt = torch.cat([tgt, next_token], dim=0)
            
          
            if next_token.item() == "</s>":
                break
                
        return tgt.squeeze(1) 