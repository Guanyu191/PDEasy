r"""Multi-Head Network type neural network models.

The Multi-Head Network type neural network models include:
    1. MHN with separate outputs in the last two layers.

TODO:
    - [ ] More MHN type models.

To create a network instance, simply pass in a list representing the number of layers 
and the number of neurons in each layer of the neural network.
In addition, you can adjust the activation function of the network and the method 
for initializing network parameters.

The default activation function of the network is Tanh. Other options include:
    1. ReLU
    2. LeakyReLU
    3. Tanh
    4. Sigmoid
    5. GELU
    6. SELU
    7. Softplus
    8. Hardtanh
    9. PReLU
    10. RReLU
    11. ELU

The default method for initializing network parameters is "xavier_normal", Other options include:
    1. kaiming_normal
    2. kaiming_uniform
    3. xavier_normal
    4. xavier_uniform
    5. normal
    6. uniform
    7. constant
    8. default

Example::
    >>> NN_LAYERS = [2, 20, 20, 20, 3]
    >>> network = MHN(NN_LAYERS)
"""
import torch
import torch.nn as nn

from pdeasy.utils.init_network_weights import init_network_weights
from pdeasy.utils.init_network_activation_function import init_network_activation_function

from typing import List, Tuple, Union


class MHN(nn.Module):
    r"""Multi-Head Network.

    It uses multiple heads for separate outputs during the output stage, 
    which is suitable for multi-output and multi-scale problems.

    Note that len(nn_layers) >= 4.

    Example::
        >>> NN_LAYERS = [2, 20, 20, 3]
        >>> network = MHN(nn_layers)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'xavier_normal',
        ):
        r"""Initialize a Multi-Head Network (MHN) instance.

        The MHN uses multiple heads for separate outputs during the output stage, 
        which is suitable for multi-output and multi-scale problems.

        Args:
            nn_layers (List[int]): A list representing the number of neurons in each layer of the neural network. 
                The length of this list should be at least 4. For example, [2, 20, 20, 3] 
                means 2 input neurons, two hidden layers with 20 neurons each, and 3 output neurons.
            act_type (str, optional): The type of activation function to be used in the network. 
                Defaults to 'tanh'. Other available options include 'ReLU', 'LeakyReLU', 'Sigmoid', 'GELU', 'SELU', 'Softplus', 'Hardtanh', 'PReLU', 'RReLU', 'ELU'.
            init_type (str, optional): The method for initializing network parameters. 
                Defaults to 'xavier_normal'. Other options include 'kaiming_normal', 'kaiming_uniform', 'xavier_uniform', 'normal', 'uniform', 'constant', 'default'.

        Attributes:
            args (dict): A dictionary storing all input arguments for reference.
            model (nn.Sequential): A sequential container representing the main body of the neural network, excluding the multi-head part.
            heads (nn.ModuleList): A module list containing multiple sequential modules, each representing a head for separate output.
        """

        super(MHN, self).__init__()
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
        }

        self.model = nn.Sequential()
        for i in range(len(nn_layers)-3):  # len(nn_layers) >= 4
            layer = nn.Sequential()
            layer.add_module(f'fc{i+1}', nn.Linear(nn_layers[i], nn_layers[i+1], bias=True))
            layer.add_module(f'act{i+1}', init_network_activation_function(act_type))
            self.model.add_module(f'layer{i+1}', layer)

        self.heads = nn.ModuleList()
        n_outputs = nn_layers[-1]
        n_hiddens = nn_layers[-2] // n_outputs
        for i in range(nn_layers[-1]):
            h = nn.Sequential()
            layer = nn.Sequential()
            layer.add_module(f'fc_head_hidden', nn.Linear(nn_layers[-3], n_hiddens, bias=True))
            layer.add_module(f'act_head_hidden', init_network_activation_function(act_type))
            h.add_module(f'layer_head_hidden', layer)
            h.add_module(f'head{i+1}', nn.Linear(n_hiddens, 1, bias=False))
            self.heads.append(h)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        X = self.model(X)

        out = []
        for h in self.heads:
            out.append(h(X))
        out = torch.cat(out, dim=1)
        return out
