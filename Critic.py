import torch.nn as nn
import torch.nn.functional as F
from Network import Network

# Critic purpose is output the approximate value of being in the current state
# Input current state, plan, reward
# Output value of current state (can be used to calc advantage and thus how good actor actions were)
# Critic backprops to Actor 1 based on reward it receives

# NOTE: Don't need to make critic network variable necessarily, mostly concerned with actor 1
# variable ability to compose actions and actor 2 predictions being variable size to scale w
# size of variable sensory input
class Critic(Network):
    def __init__(self, state_size, hidden_layer_sizes, action_size):
        super(Critic, self).__init__(state_size, hidden_layer_sizes, action_size)

        if len(hidden_layer_sizes) > 0:
            first = hidden_layer_sizes[0]
            self.linear1 = nn.Linear(self.state_size, first)
            self.linear_last = nn.Linear(hidden_layer_sizes[-1], 1)
        else:
            self.linear1 = nn.Linear(self.state_size, 1)

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
