import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        # src_key_padding_mask: True means ignore that position
        src2, attn_weights = self.self_attn(
            src, src, src, 
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attn_weights

class F1Predictor(nn.Module):
    def __init__(self, vocab_sizes, num_numeric_features, embed_dim=16, num_heads=4, num_layers=2):
        super().__init__()
        
        self.driver_emb = nn.Embedding(vocab_sizes['driverId'], embed_dim)
        self.constructor_emb = nn.Embedding(vocab_sizes['constructorId'], embed_dim)
        self.circuit_emb = nn.Embedding(vocab_sizes['circuitId'], embed_dim)
        
        self.numeric_proj = nn.Linear(num_numeric_features, embed_dim)
        
        # 3 categorical embeddings + 1 numeric projection
        self.d_model = embed_dim * 4
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(self.d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1)
        )

    def forward(self, cat_features, num_features, padding_mask=None, return_attention=False):
        d_emb = self.driver_emb(cat_features[:, :, 0])
        c_emb = self.constructor_emb(cat_features[:, :, 1])
        cir_emb = self.circuit_emb(cat_features[:, :, 2])
        
        n_proj = self.numeric_proj(num_features)
        
        x = torch.cat([d_emb, c_emb, cir_emb, n_proj], dim=-1)
        
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, src_key_padding_mask=padding_mask)
            attentions.append(attn)
            
        logits = self.output_head(x)
        
        if return_attention:
            return logits, attentions
        return logits
