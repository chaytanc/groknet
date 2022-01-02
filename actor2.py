import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Network

# Actor 2 purpose is to output the predicted current state
# Input the past state, past actions + plan
# Backprop based on actual current state and loss of last prediction

# Actor 2 prediction of next state given action and state


class Actor2(Network):
    def __init__(self, state_size, hidden_layer_sizes, action_size):
        super(Actor2, self).__init__(state_size, hidden_layer_sizes, action_size)
        self.set_input_layer_size()
        self.set_output_layer_size()
        self.init_first_last_layers(hidden_layer_sizes)
        self.make_hidden_layers(hidden_layer_sizes)

    # Override
    def set_input_layer_size(self):
        self.input_size = self.state_size + self.action_size

    # Override
    def set_output_layer_size(self):
        self.output_size = self.state_size

    def forward(self, state):
        # If no hidden layers
        if self.num_layers == 2:
            output = self.linear1(state)
        # Otherwise iterate over hidden layers and perform their transformations
        else:
            #XXX note would at some point be nice to parametrize activation funcs
            output = F.relu(self.linear1(state))
            for i in range(len(self.hidden_layers)):
                layer = self.hidden_layers[i]
                output = F.relu(layer(output))
            output = self.linear_last(output)
        return output
