import torch
import torch.nn as nn
from tqdm import tqdm
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1,  max_len,d_model)

        pe[ 0,:, 0::2] = torch.sin(position * div_term)
        pe[ 0,:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x ):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class TextClassifierTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, dim_feedforward=128,num_layers=2, output_dim=6, n_head=4, batch_first=True, dropout=0.5, max_len=5000):
        super(TextClassifierTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, source, attention_mask):
        source_embed = self.embedding(source) * math.sqrt(self.d_model)
        source_positional = self.pos_encoder(source_embed)
        z = self.transformer_encoder(src=source_positional, src_key_padding_mask=~attention_mask)
        return self.fc(z[:, 0, :])
