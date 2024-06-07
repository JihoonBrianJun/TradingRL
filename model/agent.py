import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        return (self.dropout(x)).transpose(0,1)


class StateEncoder(nn.Module):
    def __init__(self, model_dim, n_head, num_layers, src_feature_dim, window):
        super().__init__()
        
        self.src_proj = nn.Linear(src_feature_dim, model_dim)
        self.src_pos_emb = PositionalEncoding(d_model=model_dim,
                                              max_len=window)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim,
                                                        nhead=n_head,
                                                        dim_feedforward=model_dim*4,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 
                                                         num_layers=num_layers)
        
        self.out_proj1 = nn.Linear(model_dim, 1)
        self.out_proj2 = nn.Linear(window, 1)
    
    def forward(self, src, src_mask=None):
        src = self.src_pos_emb(self.src_proj(src))
        if src_mask is not None:
            src_mask = src_mask.to(src.device)
        return self.transformer_encoder(src, mask=src_mask)


class ActorNetwork(nn.Module):
    def __init__(self, action_bins, model_dim, n_head, num_layers, src_feature_dim, window):
        super().__init__()
        self.window = window
        self.model_dim = model_dim
        self.state_transformer = StateEncoder(model_dim, n_head, num_layers, src_feature_dim, window)
        self.out_proj = nn.Linear(window * model_dim, action_bins)
    
    def forward(self, src, src_mask=None):
        emb = self.state_transformer(src, src_mask)
        out = self.out_proj(emb.view(-1, self.window * self.model_dim))
        return torch.softmax(out, dim=1)


class CriticNetwork(nn.Module):
    def __init__(self, model_dim, n_head, num_layers, src_feature_dim, window):
        super().__init__()
        self.state_transformer = StateEncoder(model_dim, n_head, num_layers, src_feature_dim, window)
        self.out_proj1 = nn.Linear(model_dim, 1)
        self.out_proj2 = nn.Linear(window, 1)
    
    def forward(self, src, src_mask=None):
        emb = self.state_transformer(src, src_mask)
        return self.out_proj2(self.out_proj1(emb).squeeze(-1)).squeeze(-1)