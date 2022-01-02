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
    def __init__(self, state_size: torch.Size, hidden_size, num_layers, seq_len):
        super(TextGenerator, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.all_letters = list(string.ascii_letters + " .,;'-")
        self.all_letters.append("<eos>")
        self.int_to_char = [letter for letter in self.all_letters]
        self.hidden_size = hidden_size
        # Makes a dummy state tensor so that we can resize it and get the input_size
        # from the new dimensions
        resized_state = self.resize_state_3_dim(torch.Tensor(state_size))
        input_size = resized_state.size(-1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fully_connected = nn.Linear(hidden_size, len(self.all_letters) * seq_len)
        self.fully_connected = nn.Linear(hidden_size, len(self.all_letters))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, prev_state):
        resized = self.resize_state_3_dim(x)
        # h0, c0 = self.init_state(resized)
        output, state = self.lstm(resized, prev_state)
        output = self.fully_connected(output[:, -1, :])
        output = self.softmax(output)
        return output, state

    def init_state(self, state):
        resized = self.resize_state_3_dim(state)
        h0 = torch.zeros(self.num_layers, resized.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, resized.size(0), self.hidden_size).to(self.device)
        return h0, c0

    def recurring_forward(self, x, prev_state, string_so_far) -> str:
        # 1/seq_len % chance of ending string (geometric w lambda = 1/(1/seq_len))
        if np.random.rand() < (1 / self.seq_len):
            return string_so_far
        out, prev_state = self.forward(x, prev_state)
        char = self.output_to_ascii(out)
        string_so_far += char
        #XXX note that unless you update the weights or state in the interim, you will get the
        # same character over and over (we're not properly using the recurrence of RNN I think)
        # Want to include this probabilistic stop base case in the lstm itself's recurrence
        string_so_far = self.recurring_forward(x, prev_state, string_so_far)
        return string_so_far

    def output_to_ascii(self, output):
        # all_letters = ["<pad>"] + list(string.ascii_letters + " .,;'-") + ["<eos>"]
        ind = torch.argmax(output)
        output = self.int_to_char[ind]
        return output

    @staticmethod
    def resize_state_3_dim(state: torch.Tensor) -> torch.Tensor:
        """
        Since RNNs take 3 dim input of (batch_size, seq_len, input_size), we
        will reshape the state tensor to be 3 dim, regardless of whatever dims it had
        before. If it had more than 3 dims, squishes the ones on the end into the zeroth dimension.
        Does nothing has 3 dims
        :return: 3 dim representation of state
        """
        dims = state.dim()
        if dims < 3:
            for dim in range(3 - dims):
                state = torch.unsqueeze(state, -1)
        elif dims > 3:
            excess_dims = dims - 3
            new_size = 1
            good_dims = [state.size(x) for x in range(excess_dims, dims)]
            # Ex: if we have 5 dim state, then there are 2 bad dims; 0 and 1
            for bad_dim in range(excess_dims):
                new_size = state.size(bad_dim) * new_size
            good_dims[0] *= new_size
            state = state.reshape(*good_dims)
        return state

# Dataset:
# Training data is represented by each time we search a phrase based on what was generated, we get back the
# state, if the phrase we searched was relevant to helping predict the state and Actor1 is rewarded, so too is
# this generated text. Essentially we get training data each time this is used and don't have to label a bunch
# of words in given state as "relevant so therefore this is the target phrase to generate" or not.
# Pretrain so that it generates <eos> token?? or require somehow w .05 chance of generating at each char
# E[X = num chars generated] = 20 if p = 0.05

# Embedding maps a discrete number to a vector of numbers ie embeddings = {100: [.001, 0.3], ...}
# or one hot encoding "sheep" --> [1 0 0 0 0]


class CustomRNN(nn.LSTM):
    """
    Probabilistically quits recurring instead of trying to implement some kind of end of sequence token
    """
