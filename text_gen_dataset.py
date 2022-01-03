import torch
from torch.utils.data import Dataset, DataLoader


# batch size = 1, seq len = 20, input_size = number of sensory input measurements
# fake_state = [[1, 2, 3], [3, 0, 0]]
# Can get up to 2 items if seq_len = 1 (0 indexed), or 0 and 1 items if seq_len = 2
# get_item(0), (seq_len=2) = [[1, 3], [2, 0]]
# get_item(1), (seq_len=2) = [[2, 0], [3, 0]]
class TextGenDataset(Dataset):
    def __init__(self, state, seq_len):
        assert(seq_len <= len(state[0]))
        self.input = state
        self.seq_len = seq_len

    def __getitem__(self, item):
        seq_window_state = []
        # Last items may be useful for labeling if using RNN
        last_items = []
        # number of arrays in seq_window_state
        for i in range(self.seq_len):
            inp = []
            seq_window_state.append(inp)
        # get part of the flat sensory input we are processing for each sensory inp
        for i, val in enumerate(range(item, item + self.seq_len)):
            for sensory_input in self.input:
                feature_val = sensory_input[val]
                seq_window_state[i].append(feature_val)

            # seq_window_state.append(sensory_input[item:item + self.seq_len])
            # last_items.append(sensory_input[item + self.seq_len])
        return seq_window_state, last_items

    def __len__(self):
        return len(self.input[0]) - self.seq_len
