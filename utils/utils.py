import torch
from tensorboardX import SummaryWriter
import logging
import random
import os
import numpy as np
import networkx as nx
from datetime import datetime
from plyfile import PlyData, PlyElement


def generate_masks(data, args):
    batch_size, seq_length, _ = data.shape

    lengths = (data.sum(dim=-1) != 0).sum(dim=-1)
    pad_mask = torch.arange(args.max_seq_len).unsqueeze(0).expand(batch_size, -1) >= lengths.unsqueeze(1)
    tgt_mask = torch.triu(torch.ones(args.max_seq_len, args.max_seq_len), diagonal=1)

    return pad_mask, tgt_mask


def setup_logging(args):
    current_time = datetime.now().strftime('%m_%d_%H_%M')
    log_dir = os.path.join(args.log_dir, args.model + '_' + args.dataset + '_' + current_time)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    writer = SummaryWriter(log_dir)
    logging.info(str(args))

    return writer, log_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ply_to_graph(filename):
    vertices, edges = read_ply(filename)
    graph = nx.Graph()
    for i, vertex in enumerate(vertices):
        position = list(vertex)
        graph.add_node(i, position=position)

    for edge in edges:
        source = edge[0]
        target = edge[1]
        graph.add_edge(source, target)

    return graph


def graph_to_ply(graph, filename):
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    vertex_coordinates = np.array([graph.nodes[n]['position'] for n in nodes]).squeeze()

    if vertex_coordinates.shape[-1] == 3:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2])
            for i in range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    elif vertex_coordinates.shape[-1] == 4:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2], vertex_coordinates[i, 3])
            for i in range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'f4')])

    elif vertex_coordinates.shape[-1] == 6:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2],
             vertex_coordinates[i, 3], vertex_coordinates[i, 4], vertex_coordinates[i, 5]) for i in
            range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    elif vertex_coordinates.shape[-1] == 9:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2],
             vertex_coordinates[i, 3], vertex_coordinates[i, 4], vertex_coordinates[i, 5],
             vertex_coordinates[i, 6], vertex_coordinates[i, 7], vertex_coordinates[i, 8]) for i in
            range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                  ('l', 'f4'), ('d', 'f4'), ('c', 'f4'),
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    elif vertex_coordinates.shape[-1] == 10:
        vertex = np.array([
            (vertex_coordinates[i, 0], vertex_coordinates[i, 1], vertex_coordinates[i, 2],
             vertex_coordinates[i, 3], vertex_coordinates[i, 4], vertex_coordinates[i, 5],
             vertex_coordinates[i, 6], vertex_coordinates[i, 7], vertex_coordinates[i, 8], vertex_coordinates[i, 9]) for
            i in
            range(vertex_coordinates.shape[0])
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'f4'),
                  ('l', 'f4'), ('d', 'f4'), ('c', 'f4'),
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    edge_indices = np.array([
        (nodes.index(u), nodes.index(v)) for u, v in edges
    ], dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])

    vertex_element = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    edge_element = PlyElement.describe(edge_indices, 'edge', comments=['edge indices'])

    PlyData([vertex_element, edge_element], text=True).write(filename)


def update_metrics(metrics, batch_results, num_batches):
    for key, value in batch_results.items():
        metrics[key] = value.item()
    return {k: v / num_batches for k, v in metrics.items()}


def read_ply(filename):
    data = PlyData.read(filename)
    vertex_data = data['vertex']

    properties = vertex_data.properties

    vertices = np.array([
        tuple(vertex[prop.name] for prop in properties) for vertex in vertex_data
    ])
    edges = data['edge']

    return vertices, edges


def read_pcd(filename):
    data = PlyData.read(filename)
    vertex_data = data['vertex']

    properties = vertex_data.properties

    vertices = np.array([
        tuple(vertex[prop.name] for prop in properties) for vertex in vertex_data
    ])

    return vertices


def numpy_to_ply(matrix, ply_filename):
    matrix = matrix[matrix[:, 3] != 0]
    coords = matrix[:, :4]
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'f4')]
    G = nx.Graph()

    for i, coord in enumerate(coords):
        G.add_node(i, position=tuple(coord))

    for i in range(len(coords) - 1):
        G.add_edge(i, i + 1)

    vertex_data = np.array([tuple(coords[i]) for i in range(coords.shape[0])], dtype=dtype)
    edges = np.array([(u, v) for u, v in G.edges()], dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])

    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    edge_element = PlyElement.describe(edges, 'edge')
    PlyData([vertex_element, edge_element], text=True).write(ply_filename)


def points_to_ply(matrix, ply_filename):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_data = np.array([tuple(matrix[i]) for i in range(matrix.shape[0])], dtype=dtype)
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([vertex_element], text=True).write(ply_filename)

