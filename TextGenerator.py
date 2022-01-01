import numpy as np
import torch.nn as nn
import string
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Text_Generator purpose is to create phrases to be used as parameters of actions
# based on a given state
# Should essentially select the most relevant / interesting word(s) from the html given through
# generation of an output phrase or word
# Backprop? Don't need to, just backprop actor1 and should control this accordingly??
# or give same backprop / loss as actor 1
# Hyperparams: num_hidden_layers, seq_len, dropout of lstm
# Input: action for which we're generating text and state
# Output: phrase that acts as a parameter to the action we're taking and should improve
# Actor1 future rewards for better predictions of state

class TextGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len):
        super(TextGenerator, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.all_letters = list(string.ascii_letters + " .,;'-")
        self.int_to_char = [letter for letter in self.all_letters]
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_size, len(self.all_letters))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * x.size(0), self.hidden_size).to(self.device)
        _, output = self.lstm(x, (h0, c0))
        output = self.fully_connected(output[:, -1, :])
        output = self.softmax(output)
        return output

    def recurring_forward(self, x, string_so_far):
        # 1/seq_len % chance of ending string (geometric w lambda = 1/(1/seq_len))
        if np.random.rand() < (1 / self.seq_len):
            return string_so_far
        out = self.forward(x)
        char = self.output_to_ascii(out)
        string_so_far += char
        self.recurring_forward(x, string_so_far)

    def output_to_ascii(self, output):
        # all_letters = ["<pad>"] + list(string.ascii_letters + " .,;'-") + ["<eos>"]
        ind = torch.argmax(output)
        output = self.int_to_char[ind]
        return output

# Dataset:
# Training data is represented by each time we search a phrase based on what was generated, we get back the
# state, if the phrase we searched was relevant to helping predict the state and Actor1 is rewarded, so too is
# this generated text. Essentially we get training data each time this is used and don't have to label a bunch
# of words in given state as "relevant so therefore this is the target phrase to generate" or not.
# Pretrain so that it generates <eos> token?? or require somehow w .05 chance of generating at each char
# E[X = num chars generated] = 20 if p = 0.05
