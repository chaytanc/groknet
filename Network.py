import torch.nn as nn
import torch.nn.functional as F
from Graph import Graph

'''
General network from which to inherit abilities like creating any number of hidden layers
'''


class Network(nn.Module):
    def __init__(self, state_size, hidden_layer_sizes, action_size, hidden_layer_types=None):
        self.state_size = state_size
        self.action_size = action_size
        self.num_layers = 2 + len(hidden_layer_sizes)
        #XXX working to init these to be able to iterate through layers and get weights and biases
        self.init_first_last_layers(hidden_layer_sizes)
        self.hidden_layers = []

    def init_first_last_layers(self, hidden_layer_sizes):
        if len(hidden_layer_sizes) > 0:
            first = hidden_layer_sizes[0]
            self.linear1 = nn.Linear(self.state_size + self.action_size, first)
            self.linear_last = nn.Linear(hidden_layer_sizes[-1], self.state_size)
        else:
            self.linear1 = nn.Linear(self.state_size + self.action_size, self.state_size)

    def make_hidden_layers(self, hidden_layer_sizes):
        # Create the hidden layers if there are any
        for i in range(len(hidden_layer_sizes) - 1):
            #XXX note would at some point be nice to parametrize layer type
            # want to be able for it to evolve convolutional neural net
            hidden_layer = nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            self.hidden_layers.append(hidden_layer)

    def get_architecture(self):
        # manually get first and last layer size
        for layer in self.hidden_layers:
            pass
        for param in self.parameters():
            pass
            # if param.shape ==
            # print(self.)
