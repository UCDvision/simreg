import torch
import torch.nn as nn


def get_mlp(hidden_dims, out_dim, n_layers):
    """
    Define mlp architecture. Each 'layer' of mlp consists of a linear layer followed by batchnorm and ReLU
    non-linearity. Batchnorm and non-linearity are not added at the end of second and final linear layer.

    :param hidden_dims: list of first input dim and output dims of all layers except the final output dimension.
    e.g: if n_layers=3, hidden_dims=[ip_dim1, op_dim1, op_dim2]
    :param out_dim: dimension of output feature
    :param n_layers: number of layers to be added in the mlp
    :return: mlp (nn.Sequential() object)
    """
    layers = []
    # hidden_dims - input and output dimensions of all layers except the final output dimension
    for i in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        # no bn and relu between projection and prediction heads; if n_layers=4, proj-head=2 layers, pred-head=2 layers
        if (i != 1) and (n_layers == 4):
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden_dims[-1], out_dim))
    mlp = nn.Sequential(*layers)
    return mlp


def get_linear_proj(inp_dim, out_dim):
    """
    Define linear projection head architecture.

    :param inp_dim: dimension of input feature
    :param out_dim: dimension of output feature
    :return: mlp (nn.Sequential() object)
    """
    mlp = nn.Sequential(
        nn.BatchNorm1d(inp_dim),
        nn.Linear(inp_dim, out_dim),
    )
    return mlp


