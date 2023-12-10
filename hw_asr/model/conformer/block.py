from torch import nn

from .attention import MultiHeadedSelfAttentionModule
from .convolution import ConvolutionalModule
from .feed_forward import FeedForwardModule


class ConformerBlock(nn.Module):
    def __init__(self, embed_dim, dropout, num_heads, kernel_size):
        super().__init__()
        self.ffn1 = FeedForwardModule(embed_dim, dropout)
        self.ffn2 = FeedForwardModule(embed_dim, dropout)
        self.mhsa = MultiHeadedSelfAttentionModule(dropout, embed_dim, num_heads)
        self.conv = ConvolutionalModule(embed_dim, dropout, kernel_size)
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x):
        output = x + 1 / 2 * self.ffn1(x)
        output = output + self.mhsa(output)
        output = output + self.conv(output)
        output = self.norm(output + 1 / 2 * self.ffn2(x))
        return output
