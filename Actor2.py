import torch
import torch.nn as nn
import torch.nn.functional as F
from Network import Network

# Actor 2 purpose is to output the predicted current state
# Input the past state, past actions + plan
# Backprop based on actual current state and loss of last prediction

# Actor 2 prediction of next state given action and state

class Actor2(Network):
    def __init__(self, state_size, hidden_layer_sizes, action_size):
        super(Actor2, self).__init__(state_size, hidden_layer_sizes, action_size)
        # Check if hidden_layers exist or not and size input output layers accordingly
        if len(hidden_layer_sizes) > 0:
            first = hidden_layer_sizes[0]
            self.linear1 = nn.Linear(self.state_size + self.action_size, first)
            self.linear_last = nn.Linear(hidden_layer_sizes[-1], self.state_size)
        else:
            self.linear1 = nn.Linear(self.state_size + self.action_size, self.state_size)

        self.make_hidden_layers(hidden_layer_sizes)

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