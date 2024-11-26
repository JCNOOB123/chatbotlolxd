import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # Embeddings and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Scale embeddings by sqrt(d_model)
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        # Transform
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        
        # Project to vocabulary size
        return self.output_layer(output) 