
class Vertex:
    def __init__(self, node, bias=0):
        self.id = node
        self.bias = bias
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def update_bias(self, bias=None):
        if bias:
            self.bias = bias

    def __eq__(self, other):
        return isinstance(other, Vertex) and other.id == self.id

# useful to define architectures of networks for transfer learning
#XXX need to add locking??
class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.edge_dict = {}
        self.num_vertices = 0
        self.num_edges = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node, bias=0):
        '''
        :param node: Unique string identifier
        :param bias: Float bias of the vertex
        :return: Newly made vertex with the given identifier
        '''
        if node not in self.vert_dict:
            self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node, bias)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        return None

    def add_edge(self, edgeid, frm, to, weight=0, bias=0):
        '''
        Adds any unadded vertices and either updates or adds the edge
        Precondition: Don't add a duplicate edge or self edge
        :param edgeid:
        :param frm: vertex from
        :param to: vertex to
        :param weight: weight of the new edge
        :param bias: bias of the new edge
        :return:
        '''
        if frm == to:
            assert False

        new_edge = None

        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to, bias)
        if edgeid not in self.edge_dict:
            new_edge = Edge(edgeid, frm, to, weight)
            self.edge_dict[edgeid] = new_edge
            self.num_edges += 1
        # If you add an edge that already exists, just updates the existing edge
        else:
            self.update_edge_wb(edgeid, weight, bias)
            new_edge = self.edge_dict[edgeid]
        return new_edge

    def get_vertices(self):
        return self.vert_dict.keys()

    def get_edges(self):
        return self.edge_dict.keys()

    def update_edge_wb(self, edgeid, weight=None, bias=None):
        """
        Updates the weight of an edge or bias of the node that is the "to" node in the edge
        :param edgeid:
        :param weight:
        :param bias:
        :return:
        """
        edge = self.edge_dict[edgeid]
        edge.update_weight(weight)
        edge.to.update_bias(bias)

    def get_edge(self, edgeid):
        if edgeid in self.edge_dict:
            return self.edge_dict[edgeid]
        return None


class Edge:
    frm: str
    to: str

    def __init__(self, edgeid, frm, to, weight):
        # Each id is unique to the edge
        self.id = edgeid
        self.frm = frm
        self.to = to
        self.weight = weight

    def update_weight(self, weight=None):
        if weight:
            self.weight = weight

    def approx_same_edge(self, edge):
        """
        Returns non None / null if edge has the same id, frm, and to, but doesn't care about weight.
        :param edge: Edge to compare this one to
        :return: self if approximately the same, None otherwise
        """
        if self.id == edge.id and self.frm == edge.frm and self.to == edge.to:
            return self
        return None

# def make_net(num_layers, nodes_per_layer, connections, weightsbiases):
#         assert(len(nodes_per_layer) == num_layers)
