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
        self.set_input_layer_size()
        self.set_output_layer_size()
        # self.input_size = state_size # default val, can be overwritten
        # self.output_size = action_size # default val, can be overwritten
        self.num_layers = 2 + len(hidden_layer_sizes)
        #XXX working to init these to be able to iterate through layers and get weights and biases
        self.init_first_last_layers(hidden_layer_sizes)
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

    #XXX working here to make graph out of nn architecture -- need to do biases
    def get_architecture(self):
        g = Graph()
        # manually get first and last layer size
        for layer in self.hidden_layers:
            layer_weights = layer.weight
            # Iterate through two dimensions of connections, ex if 4 node in connects to 8 node out
            # we get size ([8, 4]) such as [[1, 2, 3, 4th col], ... 8th row]
            for next_layer_node_i, rows in enumerate(layer_weights):
                for input_layer_node_j, col_val in enumerate(rows):
                    node_from = str(input_layer_node_j)
                    node_to = str(next_layer_node_i)
                    edgeid = node_from + node_to
                    frm = g.add_vertex(node_from)
                    to = g.add_vertex(node_to)
                    g.add_edge(edgeid=edgeid, frm=frm, to=to, weight=col_val)
        for param in self.parameters():
            pass
            # if param.shape ==
            # print(self.)
