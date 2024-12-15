import torch.nn as nn


def build_rectengular_mlp(depth: int, width: int, input_size: int, output_size: int) -> nn.Module:
    """
    Build a rectangular mlp with the given depth, width, input size and output size. Works for depth >= 1.
    TODO: Turn this into a nn.Module.
    """
    layers = []
    in_features = input_size
    for _ in range(depth - 1):
        layers.extend([
            nn.Linear(in_features, width),
            nn.ReLU()
        ])
        in_features = width
    layers.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, n_hidden, norm_layer=nn.LayerNorm):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            norm_layer(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            norm_layer(n_hidden)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection


def build_residual_mlp(num_blocks: int, width: int, input_size: int, output_size: int, norm_layer=nn.BatchNorm1d) -> nn.Module:
    """
    TODO: Clean this up.
    width = n_hidden
    depth = num_blocks*2 + 2
    """
    layers = []
    layers.append(nn.Linear(input_size, width))
    layers.append(nn.ReLU())
    layers.append(norm_layer(width))
    for _ in range(num_blocks):
        layers.append(ResidualBlock(width, norm_layer))
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)
