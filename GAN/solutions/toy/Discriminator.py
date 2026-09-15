class Discriminator(nn.Module):
    """Scores a point in data space. Returns a logit, never a probability.

    Convention 1 of this lab: the last layer is a plain `nn.Linear`, and the
    probability D(x) = sigmoid(logit) is only ever formed inside a loss written with
    `softplus`, or for display. Ending the network with a sigmoid and taking the
    logarithm afterwards is the same function on paper and overflows in practice.

    The `activation` argument mirrors the generator's, and the two must be given the
    same one: an imbalance between the players is exactly what section I.10 provokes on
    purpose, and it is not something to introduce by accident here.
    """

    def __init__(self, data_dim=1, hidden=64, activation=None):
        super().__init__()
        activation = activation if activation is not None else nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden), activation,
            nn.Linear(hidden, hidden), activation,
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def probability(self, x):
        """D(x), for display only."""
        return torch.sigmoid(self.forward(x))
