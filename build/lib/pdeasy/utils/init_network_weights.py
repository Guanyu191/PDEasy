import warnings
import torch.nn as nn
import torch.nn.init as init


def init_network_weights(module, init_type='xavier_normal', init_params=None):
    if init_params is None:
        init_params = {}

    if isinstance(module, nn.Linear):
        if init_type == 'kaiming_normal':
            init.kaiming_normal_(module.weight, **init_params)
        elif init_type == 'kaiming_uniform':
            init.kaiming_uniform_(module.weight, **init_params)
        elif init_type == 'xavier_normal':
            init.xavier_normal_(module.weight, **init_params)
        elif init_type == 'xavier_uniform':
            init.xavier_uniform_(module.weight, **init_params)
        elif init_type == 'normal':
            init.normal_(module.weight, **init_params)
        elif init_type == 'uniform':
            init.uniform_(module.weight, **init_params)
        elif init_type == 'constant':
            value = init_params.get('val', 0)
            init.constant_(module.weight, value)
        else:
            warning_msg = f"Unsupported initialization type '{init_type}'. No initialization will be performed."
            warnings.warn(warning_msg, UserWarning)


if __name__ == '__main__':
    network = nn.Sequential()
    layer = nn.Sequential()
    layer.add_module('fc1', nn.Linear(2, 5, bias=True))
    layer.add_module('act1', nn.Tanh())
    network.add_module('layer1', layer)
    
    layer = nn.Sequential()
    layer.add_module('fc2', nn.Linear(5, 5, bias=True))
    layer.add_module('act2', nn.Tanh())
    network.add_module('layer2', layer)

    layer = nn.Sequential()
    layer.add_module('fc3', nn.Linear(5, 1, bias=False))
    network.add_module('layer3', layer)

    network.apply(lambda module: init_network_weights(module, 'uniform'))
    assert False
