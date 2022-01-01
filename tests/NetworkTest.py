import unittest
import torch
from Graph import Graph
from Actor2 import Actor2
from Actor1 import Actor1
from Critic import Critic
from TextGenerator import TextGenerator


class NetworkTest(unittest.TestCase):

    def setUp(self):
        state_size = 1
        action_size = 2
        hidden_layer_sizes = [1, 1]
        # should have input of state + action size
        self.atwo = Actor2(state_size, hidden_layer_sizes, action_size)
        self.aone = Actor1(state_size, hidden_layer_sizes, action_size)
        self.c = Critic(state_size, hidden_layer_sizes, action_size)
        self.seq_len = 5
        self.fake_state = torch.Tensor([1, 2, 3])
        # self.tg = TextGenerator(input_size=len(self.fake_state), hidden_size=3, num_layers=1, seq_len=self.seq_len)
        self.tg = TextGenerator(input_size=1, hidden_size=3, num_layers=3, seq_len=self.seq_len)

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
        out = self.tg.recurring_forward(self.fake_state, "")
        # Probabilistic, so may fail but should pass most the time
        assert(len(out) < 3*self.seq_len)
        assert(out != "")



if __name__ == '__main__':
    unittest.main()
