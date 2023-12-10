from torch import nn


class FeedForwardModule(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)

    def __call__(self, x):
        output = self.norm(x)
        output = self.linear1(output)
        output = self.silu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output) + x
        return output
