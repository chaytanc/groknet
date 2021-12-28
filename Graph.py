
class Vertex:
    def __init__(self, node):
        self.id = node
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


# useful to define architectures of networks for transfer learning
class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.edge_dict = {}
        self.num_vertices = 0
        self.num_edges = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        '''
        :param node: Unique string identifier
        :return: Newly made vertex with the given identifier
        '''
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
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

        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)
        if edgeid not in self.edge_dict:
            new_edge = Edge(edgeid, frm, to, weight, bias)
            self.edge_dict[edgeid] = new_edge
            self.num_edges += 1
        # If you add an edge that already exists, just updates the existing edge
        else:
            self.update_edge_wb(edgeid, weight, bias)
        return new_edge

    def get_vertices(self):
        return self.vert_dict.keys()

    def update_edge_wb(self, edgeid, weight=None, bias=None):
        self.edge_dict[edgeid].update_weight_bias(weight, bias)


class Edge:
    def __init__(self, edgeid, frm, to, weight, bias):
        # Each id is unique to the edge
        self.id = edgeid
        self.frm = frm
        self.to = to
        self.weight = weight
        self.bias = bias

    def update_weight_bias(self, weight=None, bias=None):
        if weight:
            self.weight = weight
        if bias:
            self.bias = bias

# def make_net(num_layers, nodes_per_layer, connections, weightsbiases):
#         assert(len(nodes_per_layer) == num_layers)
