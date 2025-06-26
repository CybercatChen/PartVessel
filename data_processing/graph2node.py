import os
import torch
from utils import utils
import networkx as nx
import numpy as np
import math
from plyfile import PlyElement, PlyData
import utils_process


class Node:
    def __init__(self, value, attribute, left=None, right=None, level=None, treelevel=None, maxlevel=None):
        self.left = left
        self.data = value
        self.radius = attribute
        self.right = right

        self.level = level
        self.treelevel = treelevel
        self.maxlevel = maxlevel


def graph_to_ply_with_color(graph, root_node, filename):
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    vertex_coordinates = np.array([graph.nodes[n]['position'] for n in nodes]).squeeze()
    colors = np.ones((len(nodes), 3), dtype=np.uint8) * 255  # Default to white color
    colors[root_node] = np.array([255, 0, 0], dtype=np.uint8)  # Red color for root node
    vertex = np.array([
        (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2], colors[i, 0], colors[i, 1],
         colors[i, 2])
        for i in range(vertex_coordinates.shape[0])
    ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    edge_indices = np.array([
        (nodes.index(u), nodes.index(v)) for u, v in edges
    ], dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])
    vertex_element = PlyElement.describe(vertex, 'vertex', comments=['vertices with colors'])
    edge_element = PlyElement.describe(edge_indices, 'edge', comments=['edge indices'])
    PlyData([vertex_element, edge_element], text=True).write(filename)


def find_root_node(graph, dataset):
    connected_components = list(nx.connected_components(graph))
    leaf_nodes = [node for node in graph if graph.degree(node) == 1]
    root_node = None

    def distance_from_origin(node):
        position = graph.nodes[node]['position']
        return math.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)

    if dataset == 'imagecas':
        if len(connected_components) >= 2:
            sorted_components = sorted(connected_components, key=len, reverse=True)
            component = max(sorted_components,
                            key=lambda comp: np.mean([graph.nodes[node]['position'][2] for node in comp]))
            if len(component) == 1:
                root_node = next(iter(component))
            else:
                leaf_nodes = [node for node in component if graph.degree(node) == 1]
                root_node = max(leaf_nodes, key=lambda node: graph.nodes[node]['position'][0])
            graph = utils_process.connect_root_nodes(graph)

        elif len(connected_components) < 2:
            root_node = min(leaf_nodes, key=distance_from_origin)

    elif dataset == 'march':
        root_node = min(leaf_nodes, key=distance_from_origin)

    elif dataset == 'intra':
        target = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
        root_node = min(leaf_nodes, key=lambda node: np.linalg.norm(
            np.array(graph.nodes[node]['position'][:3], dtype=np.float64) - target))

    elif dataset == 'cow':
        root_node = min(leaf_nodes, key=lambda node: graph.nodes[node]['position'][2])

    return root_node, graph


def graph_to_node(graph, dataset):
    root_node, graph = find_root_node(graph, dataset)
    dfs_tree = nx.dfs_tree(graph, source=root_node)
    max_level = nx.dag_longest_path_length(dfs_tree)
    total_level = sum(nx.shortest_path_length(dfs_tree, root_node).values())

    nodes_dict = {}
    for node in dfs_tree.nodes():
        position = graph.nodes[node]['position']
        level = nx.shortest_path_length(dfs_tree, root_node, node)
        nodes_dict[node] = Node(
            value=node,
            attribute=torch.tensor(position, dtype=torch.float, device='cuda'),
            level=level,
            treelevel=total_level,
            maxlevel=max_level)

    for node in dfs_tree.nodes():
        children = list(dfs_tree.successors(node))
        if len(children) == 2:
            parent_pos = np.array(graph.nodes[node]['position'])[:3]
            child1_pos = np.array(graph.nodes[children[0]]['position'])[:3]
            child2_pos = np.array(graph.nodes[children[1]]['position'])[:3]

            vec1 = child1_pos - parent_pos
            vec2 = child2_pos - parent_pos
            cross_y = np.cross(vec1, vec2)[1]

            if cross_y > 0:  # child1 is left of child2
                nodes_dict[node].left = nodes_dict[children[0]]
                nodes_dict[node].right = nodes_dict[children[1]]
            else:  # child1 is right of child2
                nodes_dict[node].left = nodes_dict[children[1]]
                nodes_dict[node].right = nodes_dict[children[0]]

        elif len(children) == 1:
            nodes_dict[node].left = nodes_dict[children[0]]

    return nodes_dict[root_node], len(graph), root_node


def save_key_graph(data_path, output_path, dataset_name):
    trees = []
    graphs = []
    num_nodes = []
    file_names = []
    max_nodes = 0
    os.makedirs(os.path.join(output_path, 'node'), exist_ok=True)

    for file in os.listdir(data_path):
        data_file = os.path.join(data_path, file)
        graph = utils.ply_to_graph(data_file)
        graphs.append(graph)
        file_names.append(file)
        root, num_node, root_node = graph_to_node(graph, dataset_name)
        filename = os.path.join(output_path, 'node', f'{file}.ply')
        graph_to_ply_with_color(graph, root_node, filename)
        max_nodes = max(max_nodes, num_node)
        trees.append(root)
        num_nodes.append(num_node)

    all_data = {
        'data': trees,
        'graphs': graphs,
        'num_nodes': num_nodes,
        'file_names': file_names,
    }

    output_file_path = os.path.join(output_path, f'{dataset_name}.pt')
    torch.save(all_data, output_file_path)

    print(f"Saved all data to: {output_file_path}")
