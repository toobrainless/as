from torch import nn


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, dropout, embed_dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        output = self.norm(x)
        output = self.attention(output, output, output, need_weights=False)[0]
        output = self.dropout(output) + x
        return output
