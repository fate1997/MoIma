from torch import nn


ACTIVATION_REGISTER = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'none': nn.Identity()
}


def get_activation(activation: str):
    activation = activation.lower()
    if activation not in ACTIVATION_REGISTER:
        raise ValueError(f'Activation function {activation} not supported.')
    return ACTIVATION_REGISTER[activation]