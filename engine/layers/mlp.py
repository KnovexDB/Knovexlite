from torch import nn

class MLP(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_size, num_hidden_layers
    ):
        super(MLP, self).__init__()
        layers = []
        in_channels = in_channels
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_channels, hidden_size))
            layers.append(nn.ReLU())
            in_channels = hidden_size
        layers.append(nn.Linear(hidden_size, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)