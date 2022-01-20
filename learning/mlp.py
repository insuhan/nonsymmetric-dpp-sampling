import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[], dropout=0, activation="selu", batchnorm=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.batchnorm = batchnorm

        if activation is not None:
            if activation.lower() == "relu":
                activation = nn.ReLU
            elif activation.lower() == "selu":
                activation = nn.SELU
            elif activation.lower() == "tanh":
                activation = nn.Tanh
            elif activation.lower() == "sigmoid":
                activation = nn.Sigmoid
            else:
                raise NotImplementedError(activation)

        layers = []
        layer_dims = [input_dim] + list(hidden_dims) + [output_dim]
        num_layers = len(layer_dims)
        for h in range(num_layers - 1):
            in_dim, out_dim = layer_dims[h], layer_dims[h + 1]

            if batchnorm:
                layers.append(nn.BatchNorm1d(in_dim))
            layers.append(nn.Linear(in_dim, out_dim))

            if h != num_layers - 2:
                if activation is not None:
                    layers.append(activation())
                    if dropout:
                        layers.append(nn.AlphaDropout(p=dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        z = x
        batch_size = x.size(0)
        for layer in self.layers:
            if batch_size == 1 and isinstance(layer, nn.BatchNorm1d):
                continue
            z = layer(z)
        return z
