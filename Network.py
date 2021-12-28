import torch.nn as nn
import torch.nn.functional as F

'''
General network from which to inherit abilities like creating any number of hidden layers
'''


class Network(nn.Module):
    def __init__(self, state_size, hidden_layer_sizes, action_size, hidden_layer_types=None):
        self.state_size = state_size
        self.action_size = action_size
        self.num_layers = 2 + len(hidden_layer_sizes)
        self.hidden_layers = []

    def make_hidden_layers(self, hidden_layer_sizes):
        # Create the hidden layers if there are any
        for i in range(len(hidden_layer_sizes) - 1):
            #XXX note would at some point be nice to parametrize layer type
            # want to be able for it to evolve convolutional neural net
            hidden_layer = nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            self.hidden_layers.append(hidden_layer)
