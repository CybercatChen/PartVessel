import os

from utils.utils import graph_to_ply, ply_to_graph
import utils_assembly


def assemble(tree_graphs, seq_args, assembly_args):
    model_seq = utils_assembly.load_model(tree_args=None, seq_args=seq_args, assembly_args=assembly_args)
    output_path = utils_assembly.create_result_folder(assembly_args, 'recon')

    for filename in os.listdir(input_folder):
        if filename.endswith('.ply'):
            input_path = os.path.join(input_folder, filename)
            tree_graph = ply_to_graph(input_path)
            skeleton = utils_assembly.traverse_and_paste_curves(tree_graph, model_seq, seq_args, assembly_args.dataset,
                                                                target_distance=0.01)
            skeleton = utils_assembly.smooth_bifurcation_node(skeleton, iterations=10, smooth_factor=0.3)
            if skeleton.number_of_nodes() == 0:
                continue
            graph_to_ply(skeleton, os.path.join(output_path, f'skeleton_{filename}'))
    return tree_graphs


if __name__ == '__main__':
    from config import tree_args, seq_args
    import argparse

    parser = argparse.ArgumentParser(description="ASSEMBLY Arguments")
    parser.add_argument('--dataset', type=str, default='march', choices=['imagecas', 'march', 'cow'])
    parser.add_argument('--gen_num', type=int, default=50, help="Number of samples to generate")
    parser.add_argument('--mmd_distance', type=str, default='rbf', help="Type of MMD distance (e.g., 'rbf')")
    parser.add_argument('--max_subgraph', type=bool, default=True, help="Whether to use maximum subgraph")
    # File Paths
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--model_name', type=str, default=r'')
    parser.add_argument('--seq_path', type=str, default=r'')
    parser.add_argument('--max_seq_len', type=int, default=200, help="Maximum sequence length")

    assembly_arg = parser.parse_args()

    tree_arg = tree_args()
    seq_arg = seq_args()
    input_folder = r''
    assemble(input_folder, seq_arg, assembly_arg)
