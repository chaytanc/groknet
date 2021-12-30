import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from Graph import Graph

'''
General network from which to inherit abilities like creating any number of hidden layers
'''


class Network(nn.Module):
    linear_last: Linear
    linear1: Linear

    def __init__(self, state_size, hidden_layer_sizes, action_size, hidden_layer_types=None):
        super(Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.set_input_layer_size()
        self.set_output_layer_size()
        # self.input_size = state_size # default val, can be overwritten
        # self.output_size = action_size # default val, can be overwritten
        self.num_layers = 2 + len(hidden_layer_sizes)
        #XXX working to init these to be able to iterate through layers and get weights and biases
        # self.init_first_last_layers(hidden_layer_sizes)
        self.hidden_layers = []

    def set_input_layer_size(self):
        raise NotImplementedError("Must override this method")

    def set_output_layer_size(self):
        raise NotImplementedError("Must override this method")

    # Want guaranteed first and last layer explicitly defined for all Network classes
    # so we call it in the init but can override and recall for different input and output sizes
    def init_first_last_layers(self, hidden_layer_sizes):
        if len(hidden_layer_sizes) > 0:
            first = hidden_layer_sizes[0]
            self.linear1 = nn.Linear(self.input_size, first)
            self.linear_last = nn.Linear(hidden_layer_sizes[-1], self.output_size)
        else:
            self.linear1 = nn.Linear(self.input_size, self.output_size)

    def make_hidden_layers(self, hidden_layer_sizes):
        # Create the hidden layers if there are any
        for i in range(len(hidden_layer_sizes) - 1):
            #XXX note would at some point be nice to parametrize layer type
            # want to be able for it to evolve convolutional neural net
            hidden_layer = nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            self.hidden_layers.append(hidden_layer)

    #XXX working here to make graph out of nn architecture -- need to unit test! can grab some from
    # old graph tests maybe
    # First layer has no biases but each other layer has a matrix of biases added to each incoming
    # computation
    def get_architecture(self):
        g = Graph()
        # manually get first and last layer size
        first_biases = np.zeros(self.input_size)
        first_weights = self.linear1.weight
        last_biases = self.linear_last.bias
        last_weights = self.linear_last.weight
        self.construct_graph_from_layer(g, first_weights, first_biases, 0)
        # -1 represents last layer number
        last_layer_num = len(self.hidden_layers) + 1
        self.construct_graph_from_layer(g, last_weights, last_biases, last_layer_num)

        for layer_num, layer in enumerate(self.hidden_layers):
            layer_weights = layer.weight
            layer_biases = layer.bias
            self.construct_graph_from_layer(g, layer_weights, layer_biases, layer_num + 1)
        return g

    @staticmethod
    def construct_graph_from_layer(g, layer_weights, layer_biases, layer_num):
        # Iterate through two dimensions of connections, ex if 4 node in connects to 8 node out
        # we get size ([8, 4]) ([output, input]) such as [[1, 2, 3, 4th col], ... 8th row]
        for next_layer_node_i, rows in enumerate(layer_weights):
            bias = layer_biases[next_layer_node_i]
            for input_layer_node_j, col_val in enumerate(rows):
                # if layer_num == -1 :
                #     node_from = str(input_layer_node_j) + "-" + str(len())
                #     node_to = str(next_layer_node_i) + "-" + str(layer_num + 1)
                #     edgeid = str(node_from + "_" + node_to)
                # else:
                node_from = str(input_layer_node_j) + "-" + str(layer_num)
                node_to = str(next_layer_node_i) + "-" + str(layer_num + 1)
                edgeid = str(node_from + "_" + node_to)
                g.add_edge(edgeid=edgeid, frm=node_from, to=node_to, weight=col_val, bias=bias)
