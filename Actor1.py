import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from Network import Network
import actions

# Actor 1 purpose is to output next real action and plan size or to be used for getting the
# next action to add to the plan and maintain plan size?? while adding to the plan
# Input the current or predicted state s
# Critic backprops based on the reward it receives



# Actor 1 returns action in form of a Plan
# input layer is parameter (so that evolution of generations can tune) and
# is initially size of


class Actor1(Network):
    def __init__(self, state_size, hidden_layer_sizes, action_size):
        super(Actor1, self).__init__(state_size, hidden_layer_sizes, action_size)
        # Check if hidden_layers exist or not and size input output layers accordingly
        if len(hidden_layer_sizes) > 0:
            first = hidden_layer_sizes[0]
            self.linear1 = nn.Linear(self.state_size, first)
            self.plan_linear1 = self.linear1
            self.linear_last = nn.Linear(hidden_layer_sizes[-1], self.action_size)
            self.plan_last = nn.Linear(hidden_layer_sizes[-1], 1)
        else:
            self.linear1 = nn.Linear(self.state_size, self.action_size)
            self.plan_linear1 = nn.Linear(self.state_size, 1)

        # Create the hidden layers if there are any
        self.make_hidden_layers(hidden_layer_sizes)

    def forward(self, state):
        # If no hidden layers
        if(self.num_layers == 2):
            output = self.linear1(state)
            plan_output = self.plan_linear1(state)
        # Otherwise iterate over hidden layers and perform their transformations
        else:
            output = F.relu(self.linear1(state))
            plan_output = F.relu(self.plan_linear1(state))
            for i in range(len(self.hidden_layers)):
                layer = self.hidden_layers[i]
                output = F.relu(layer(output))
                plan_output = F.relu(layer(plan_output))
            output = self.linear_last(output)
            plan_output = self.plan_last(plan_output)
        distribution = Categorical(F.softmax(output, dim=-1))
        plan_size = int(plan_output) #XXX not sure this works to get a good plan size...?
        return distribution, plan_size

    # call forward and then turn distribution into an action
    # return action as a function to take
    def get_action(self, state):
        dist, plan_size = self.forward(state)
        ind_of_highest = torch.argmax(dist)
        acts, num_actions = actions.get_actions_num_actions(actions.Actions)
        assert(len(dist) == num_actions)
        name, action = acts[ind_of_highest]
        return action


    # NOTE can scrap plan buffer for now and just use iterations_before_backprop param to make
    # scheduling properties / sequential action learning
    # Then don't have to implement interrupt scheduler to deal with buffer as well
    def execute_plan(self):
        # Add action to self.plan, return first action in plan??
        # While self.plan is not empty, call get_action
        pass


# Merge actions and prediction func
