import unittest
import torch
from torch.utils.data import DataLoader

from graph import Graph
from actor2 import Actor2
from actor1 import Actor1
from critic import Critic
from text_gen_dataset import TextGenDataset
from text_generator import TextGenerator


class NetworkTest(unittest.TestCase):

    def setUp(self):
        state_size = 1
        action_size = 2
        hidden_layer_sizes = [1, 1]
        # should have input of state + action size
        self.atwo = Actor2(state_size, hidden_layer_sizes, action_size)
        self.aone = Actor1(state_size, hidden_layer_sizes, action_size)
        self.c = Critic(state_size, hidden_layer_sizes, action_size)
        self.seq_len = 2
        self.fake_state = [[1, 2, 3], [3, 0, 0], [4, 4, 9]]
        self.state_data = TextGenDataset(state=self.fake_state, seq_len=self.seq_len)
        self.data: torch.Tensor = torch.Tensor([x for x, _ in self.state_data])
        # self.tg = TextGenerator(input_size=len(self.fake_state), hidden_size=3, num_layers=1, seq_len=self.seq_len)
        self.tg = TextGenerator(empty_dataset_state=self.data, hidden_size=3, num_layers=3, seq_len=self.seq_len)

    # Example
    # fake_state = [[1, 2, 3], [3, 0, 0]]
    # Can get up to ind=2 items if seq_len = 1 (0 indexed), or 0 and 1 items if seq_len = 2
    # get_item(0), (seq_len=2) = [[1, 3], [2, 0]]
    # get_item(1), (seq_len=2) = [[2, 0], [3, 0]]
    def test_text_gen_dataset(self):
        # size of batch is just num of seqs we can make from our input, which will be num of predictions we make
        # because we want to use entire sequence for each prediction
        # If we have three observations and use seq len of 2 we can make two predictions, or observations - seq_len
        # state_data = [batch=[ seqs=[ inputs=[1, 3, 4], [2, 0, 4]], [[2, 0, 4], [3, 0, 9]] ]
        data: torch.Tensor = torch.Tensor([x for x, _ in self.state_data])

        # batch size = 2, seq_len = 2, input size = num of sensory observations composing a state = 3
        # batch number = 1 per state
        assert(data.dim() == 3)
        assert(data.size() == torch.Size([2, 2, 3]))

    def test_get_architecture(self):
        """
        Tests that the graph made from the pytorch nn.Module object layers is accurate
        """
        g1 = self.aone.get_architecture()
        g2 = self.atwo.get_architecture()
        c = self.c.get_architecture()
        # Since weights and biases randomly init for pytorch, won't verify that they match
        aone_graph = Graph()
        atwo_graph = Graph()
        c_graph = Graph()
        e0 = aone_graph.add_edge("0-0_0-1", "0-0", "0-1", 0, 0)
        e1 = aone_graph.add_edge("0-1_0-2", "0-1", "0-2", 0, 0)
        e2 = aone_graph.add_edge("0-2_0-3", "0-2", "0-3", 0, 0)
        e3 = aone_graph.add_edge("0-2_1-3", "0-2", "1-3", 0, 0)

        atwo_graph.add_edge("0-0_0-1", "0-0", "0-1", 0, 0)
        atwo_graph.add_edge("1-0_0-1", "1-0", "0-1", 0, 0)
        atwo_graph.add_edge("2-0_0-1", "2-0", "0-1", 0, 0)
        atwo_graph.add_edge("0-1_0-2", "0-1", "0-2", 0, 0)
        atwo_graph.add_edge("0-2_0-3", "0-2", "0-3", 0, 0)

        c_graph.add_edge("0-0_0-1", "0-0", "0-1", 0, 0)
        c_graph.add_edge("0-1_0-2", "0-1", "0-2", 0, 0)
        c_graph.add_edge("0-2_0-3", "0-2", "0-3", 0, 0)
        for edgeid, edge in aone_graph.edge_dict.items():
            actual_edge = g1.get_edge(edgeid)
            assert actual_edge.approx_same_edge(edge)

        for edgeid, edge in atwo_graph.edge_dict.items():
            actual_edge = g2.get_edge(edgeid)
            assert actual_edge
            assert actual_edge.approx_same_edge(edge)

        for edgeid, edge in c_graph.edge_dict.items():
            actual_edge = c.get_edge(edgeid)
            assert actual_edge.approx_same_edge(edge)

    def test_text_gen(self):
        full = ''
        init_state = (self.tg.init_state(self.fake_state))
        # out = self.tg.recurring_forward(self.fake_state, "")
        # out = self.tg.recurring_forward(self.fake_state, init_state, "")
        # char, hidden = self.tg.forward(self.data, init_state)
        out, hidden = self.tg(self.data, init_state)
        char = self.tg.output_to_ascii(out)
        #XXX currently generates the same char over and over
        while char != '<eos>':
            full += char
            out, hidden = self.tg(self.data, hidden)
            char = self.tg.output_to_ascii(out)

        # Probabilistic, so may fail but should pass most the time
        assert(len(full) < len(self.state_data))
        print("full: ", full)
        assert(full != "")
    #
    # def test_text_gen_resize(self):
    #     # fake states:
    #     # total_dims = 2*3 = 6
    #     too_small = torch.Tensor(2, 3)
    #     # total dims = 2*3*4*5 = 120
    #     too_big = torch.Tensor(2, 3, 4, 5)
    #     really_big = torch.Tensor(2, 3, 4, 5, 2, 2)
    #     new = self.tg.resize_state_3_dim(too_small)
    #     assert(new.dim() == 3)
    #     assert(new.size() == torch.Size([2, 3, 1]))
    #     new = self.tg.resize_state_3_dim(too_big)
    #     assert(new.dim() == 3)
    #     assert(new.size() == torch.Size([6, 4, 5]))
    #     new = self.tg.resize_state_3_dim(really_big)
    #     assert(new.dim() == 3)
    #     assert(new.size() == torch.Size([120, 2, 2]))

    def tearDown(self) -> None:
        self.aone.actions_set.end()
        # self.atwo.actions_set.end()
        # self.c.actions_set.end()



if __name__ == '__main__':
    unittest.main()
