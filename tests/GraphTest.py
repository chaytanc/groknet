import unittest
from graph import Graph


class GraphTest(unittest.TestCase):

    # // Sets up a full graph used for tests
    def setUp(self):
        # Graph:
        # a b
        # \/
        # c
        # /\
        # d e

        # Graphs
        self.graph = Graph()
        self.emptyGraph = Graph()
        # Nodes
        self.aNode = self.graph.add_vertex("a")
        self.bNode = self.graph.add_vertex("b")
        self.cNode = self.graph.add_vertex("c")
        self.dNode = self.graph.add_vertex("d")
        self.eNode = self.graph.add_vertex("e")
        # XXX make fake layers to test in style of tensor
        # Edges
        self.edgeAC = self.graph.add_edge("1", "a", "c", weight=1, bias=1)
        self.edgeBC = self.graph.add_edge("2", "b", "c", 1, 1)
        self.edgeCD = self.graph.add_edge("3", "c", "d", 1, 1)
        self.edgeCE = self.graph.add_edge("4", "c", "e", 1, 1)

    def test_add_edge(self):
        edge = self.emptyGraph.add_edge("nonexistent vertices edge", "a", "b", 3, 4)
        assert (self.emptyGraph.get_edge("nonexistent vertices edge") == edge)
        a = self.emptyGraph.get_vertex("a")
        b = self.emptyGraph.get_vertex("b")
        assert (a and b)
        assert (edge.weight == 3 and a.bias == 0 and b.bias == 4)
        # Add a redundant edge
        edge = self.emptyGraph.add_edge("nonexistent vertices edge", "a", "b", 5, 6)
        a = self.emptyGraph.get_vertex("a")
        b = self.emptyGraph.get_vertex("b")
        assert (a is not None and b is not None)
        assert (self.emptyGraph.get_edge("nonexistent vertices edge") == edge)
        assert (edge.weight == 5 and a.bias == 0 and b.bias == 6)

    def test_get_vertex(self):
        a = self.graph.get_vertex("a")
        b = self.graph.get_vertex("b")
        assert a and b
        self.graph.add_vertex("a")
        a = self.graph.get_vertex("a")
        assert a

    def test_add_layer(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
