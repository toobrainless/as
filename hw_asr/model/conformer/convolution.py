from torch import nn


class ConvolutionalModule(nn.Module):
    """
    https://towardsdatascience.com/efficient-image-segmentation-using-pytorch-part-3-3534cf04fb89#:~:text=A%20depthwise%20grouped%20convolution%2C%20where,called%20a%20%E2%80%9Cgrouped%E2%80%9D%20convolution.
    """

    def __init__(self, embed_dim, dropout, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1, "for mathcing in and out dim of depthwise conv"
        self.norm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size,
            groups=embed_dim,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(embed_dim)
        self.silu = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        output = self.norm(x)
        output = output.transpose(-1, -2)
        output = self.pointwise_conv1(output)
        output = self.glu(output)
        output = self.depthwise_conv(output)
        output = self.bn(output)
        output = self.silu(output)
        output = self.pointwise_conv2(output)
        output = self.dropout(output).transpose(-1, -2) + x
        return output
