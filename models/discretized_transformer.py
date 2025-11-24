import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiscretizedMultiHeadSelfAttention(nn.Module):
    """
    Gumbel-Softmaxを使ってAttention重みを離散化するMulti-Head Self-Attention
    """
    def __init__(self, d_model, n_heads, dropout=0.1, temperature=1.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        Gumbel-Softmax sampling
        Args:
            logits: (*, n_classes)
            temperature: float
            hard: bool - if True, return one-hot vector
        Returns:
            y: (*, n_classes)
        """
        if self.training:
            # Add Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
        else:
            y = F.softmax(logits / temperature, dim=-1)
        
        if hard:
            # Straight-through estimator
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        
        return y
        
    def forward(self, x, mask=None, hard=False):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) or None
            hard: bool - if True, use hard discretization
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, n_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Gumbel-SoftmaxをAttention重み
        attn_weights = self.gumbel_softmax(scores, self.temperature, hard=hard)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V) 
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        
        return out


class DiscretizedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, temperature=1.0):
        super().__init__()
        
        self.attention = DiscretizedMultiHeadSelfAttention(d_model, n_heads, dropout, temperature)
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
        
    def forward(self, x, mask=None, hard=False):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) or None
            hard: bool - if True, use hard discretization
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, mask, hard)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class DiscretizedTransformer(nn.Module):
    def __init__(self, nchar, d_model=128, n_heads=4, n_layers=2, d_ff=512, 
                 dropout=0.1, temperature=1.0, max_len=5000):
        super().__init__()
        
        self.nchar = nchar
        self.d_model = d_model
        self.temperature = temperature
        self.embedding = nn.Embedding(nchar, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            DiscretizedTransformerBlock(d_model, n_heads, d_ff, dropout, temperature)
            for _ in range(n_layers)
        ])
        
        # Output layer
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
    
    def set_temperature(self, temperature):
        """温度パラメータを設定"""
        self.temperature = temperature
        for layer in self.layers:
            layer.attention.temperature = temperature
    
    def forward(self, x, hard=False):
        """
        Args:
            x: (batch_size, seq_len) - token indices
            hard: bool - if True, use hard discretization
        Returns:
            out: (batch_size, seq_len, nchar) - logits for each position
        """
        batch_size, seq_len = x.size()
        
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model) 
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)

        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  
        
        for layer in self.layers:
            x = layer(x, mask, hard)
    
        out = self.fc_out(x)
        
        return out
