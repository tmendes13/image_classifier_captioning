"""Image classifiers for Part 2."""

from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    """Map flattened images to logits. Empty hidden_sizes gives the linear baseline."""

    def __init__(self, in_dim=3 * 64 * 64, num_classes=10, hidden_sizes=(), activation="relu",
                 dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        width = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @property
    def first_layer(self):
        """The first nn.Linear (its weights are visualized as images in Q2.4)."""
        return self.net[1]
