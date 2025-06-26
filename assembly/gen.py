import os

import torch

from utils.utils import graph_to_ply
from test_tree import decode_testing
import utils_assembly


def tree_sample(model_tree, tree_args, output_path, num_sample):
    model_tree.eval()
    graphs = []
    for i in range(num_sample):
        vector = torch.randn(1, tree_args.latent_size).to('cuda')
        gen_tree = decode_testing(vector, max=30, decoder=model_tree)
        gen_graph = gen_tree.to_graph(dec=True)
        graph_to_ply(gen_graph, os.path.join(output_path, f'gen_tree_{i}.ply'))
        graphs.append(gen_graph)
    return graphs


def gen(tree_args, seq_args, assembly_args):
    model_tree, model_seq = utils_assembly.load_model(tree_args, seq_args, assembly_args)
    output_path = utils_assembly.create_result_folder(assembly_args, 'gen')
    tree_graphs = tree_sample(model_tree, tree_args, output_path, assembly_args.gen_num)
    for i, graph in enumerate(tree_graphs):
        if assembly_args.dataset == 'imagecas':
            graph = utils_assembly.remove_edge(graph)
        skeleton = utils_assembly.traverse_and_paste_curves(graph, model_seq, seq_args, assembly_args.dataset, target_distance=0.05)
        smoothed_skeleton_graph = utils_assembly.smooth_bifurcation_node(skeleton, iterations=10, smooth_factor=0.3)
        graph_to_ply(smoothed_skeleton_graph, os.path.join(output_path, f'gen_skeleton_{i}.ply'))
    return tree_graphs


if __name__ == '__main__':
    from config import tree_args, seq_args
    import argparse

    parser = argparse.ArgumentParser(description="ASSEMBLY Arguments")
    parser.add_argument('--dataset', type=str, default='imagecas', choices=['imagecas', 'march', 'cow'])
    parser.add_argument('--gen_num', type=int, default=100, help="Number of samples to generate")
    parser.add_argument('--mmd_distance', type=str, default='rbf', help="Type of MMD distance (e.g., 'rbf')")
    parser.add_argument('--max_subgraph', type=bool, default=True, help="Whether to use maximum subgraph")
    # File Paths
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--model_name', type=str, default=r'')

    parser.add_argument('--seq_path', type=str, default=r'')
    parser.add_argument('--tree_path', type=str,
                        default=r'')
    parser.add_argument('--max_seq_len', type=int, default=200, help="Maximum sequence length")

    assembly_arg = parser.parse_args()
    tree_arg = tree_args()
    seq_arg = seq_args()

    gen(tree_arg, seq_arg, assembly_arg)
