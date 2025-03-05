import torch.nn as nn


class RectangularMLP(nn.Module):
    """
    A rectangular MLP with configurable depth, width, input size and output size. Works for depth >= 1.
    """
    def __init__(self, depth: int, width: int, input_size: int, output_size: int):
        super(RectangularMLP, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(in_features, width),
                nn.ReLU()
            ])
            in_features = width
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
        return x + self.block(x)


class ResidualMLP(nn.Module):
    """
    width = n_hidden
    depth = num_blocks*2 + 2
    """
    def __init__(self, num_blocks: int, width: int, input_size: int, output_size: int, norm_layer=nn.BatchNorm1d):
        super(ResidualMLP, self).__init__()
        layers = [
            nn.Linear(input_size, width),
            nn.ReLU(),
            norm_layer(width),
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(width, norm_layer))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
