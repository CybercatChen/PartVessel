import networkx as nx


def count_fn(f):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return f(*args, **kwargs)

    wrapper.count = 0
    return wrapper


@count_fn
def create_node(data, radius, left=None, right=None):
    return Node(data, radius, left, right)


class Node:
    def __init__(self, value, radius, left=None, right=None, level=None, treelevel=None, maxlevel=None):
        self.left = left
        self.data = value
        self.radius = radius
        self.right = right
        self.level = level

        self.treelevel = treelevel
        self.maxlevel = maxlevel

    def to_gpu(self):
        self.radius = self.radius.cuda()
        if self.left:
            self.left.to_gpu()
        if self.right:
            self.right.to_gpu()

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_two_child(self):
        return self.left is not None and self.right is not None

    def is_one_child(self):
        return (self.left is None) != (self.right is None)

    def child_num(self):
        return sum(child is not None for child in (self.left, self.right))

    def height(self, root):
        if root is None:
            return 0
        leftAns = self.height(root.left)
        rightAns = self.height(root.right)

        return max(leftAns, rightAns) + 1

    def to_graph(self, dec=False):
        graph = nx.Graph()
        self.add_node(graph, dec)
        return graph

    def add_node(self, graph, dec=False):
        radius = self.radius.cpu().detach().numpy()
        if dec:
            radius = radius[0]
        graph.add_node(self.data, position=radius, radius=radius)

        if self.left:
            graph.add_edge(self.data, self.left.data)
            self.left.add_node(graph, dec)

        if self.right:
            graph.add_edge(self.data, self.right.data)
            self.right.add_node(graph, dec)

    def Tree2Graph(self, graph):
        radius = self.radius
        coordinate = self.radius
        graph.add_node(self.data + 1, position=coordinate, radius=radius)

        if self.right is not None:
            self.right.Tree2Graph(graph)
            graph.add_edge(self.data + 1, self.right.data + 1)

        if self.left is not None:
            self.left.Tree2Graph(graph)
            graph.add_edge(self.data + 1, self.left.data + 1)

        else:
            return graph
