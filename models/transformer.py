import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, n_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V) 
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) or None
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class Transformer(nn.Module):
    def __init__(self, nchar, d_model=128, n_heads=4, n_layers=2, d_ff=512, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.nchar = nchar
        self.d_model = d_model
        
        self.embedding = nn.Embedding(nchar, d_model)
        
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)      
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        

        self.fc_out = nn.Linear(d_model, nchar)
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding matrix"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - token indices
        Returns:
            out: (batch_size, seq_len, nchar) - logits for each position
        """
        batch_size, seq_len = x.size()
        
        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        out = self.fc_out(x)  # (batch_size, seq_len, nchar)
        
        return out