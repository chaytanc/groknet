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
        self.set_input_layer_size()
        self.set_output_layer_size()
        self.init_first_last_layers(hidden_layer_sizes)
        self.make_hidden_layers(hidden_layer_sizes)

    # Override
    def set_input_layer_size(self):
        self.input_size = self.state_size

    # Override
    def set_output_layer_size(self):
        self.output_size = 1

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
