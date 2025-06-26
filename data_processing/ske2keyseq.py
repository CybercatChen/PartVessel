import os
import numpy as np
import networkx as nx
import torch
import utils_process
from utils import utils


def get_key_graph(key_graph_folder, skeleton_folder, label_folder,
                  radius_folder, output_path, max_seq_len=None, datasetname='imagecas'):
    all_sequences = []
    all_conditions = []
    all_lengths = []
    all_filenames = []

    output_seq_path = os.path.join(output_path, 'segments')
    output_key_path = os.path.join(output_path, 'key')
    os.makedirs(output_seq_path, exist_ok=True)
    os.makedirs(output_key_path, exist_ok=True)

    for skeleton_file in os.listdir(skeleton_folder):
        if skeleton_file.endswith('.ply'):
            patient_id = skeleton_file.split('.')[0]
            skeleton_path = os.path.join(skeleton_folder, skeleton_file)
            label_path = os.path.join(label_folder, patient_id + '.ply.skeleton.seg')
            key_graph_path = os.path.join(key_graph_folder, patient_id + '.ply.skeleton.ply')
            radius_path = os.path.join(radius_folder, patient_id + '.ply.radius')

            skeleton = utils.ply_to_graph(skeleton_path)
            labels = np.loadtxt(label_path)
            radii = np.loadtxt(radius_path)
            key = utils.ply_to_graph(key_graph_path)

            skeleton_comp = list(nx.connected_components(skeleton))
            skeleton_cc = max(skeleton_comp, key=len)
            skeleton_graph = skeleton.subgraph(skeleton_cc).copy()

            key_comp = list(nx.connected_components(key))
            key_cc = max(key_comp, key=len)
            key_graph = key.subgraph(key_cc).copy()

            for key_node in list(key_graph.nodes()):
                key_pos = key_graph.nodes[key_node]['position'][:3]
                skeleton_node = utils_process.find_skeleton_node_by_position(skeleton_graph, key_pos)
                label = labels[skeleton_node]
                segment_indices = np.where(labels == label)[0]
                segment_subgraph = skeleton_graph.subgraph(segment_indices).copy()

                if segment_subgraph.number_of_nodes() <= 5:
                    if key_graph.degree[key_node] == 1:
                        connected_edges = list(key_graph.edges(key_node))
                        if connected_edges:
                            for edge in connected_edges:
                                key_graph.remove_edge(*edge)
                        key_graph.remove_node(key_node)
                        continue

                if not nx.is_connected(segment_subgraph):
                    skeleton_comp = list(nx.connected_components(segment_subgraph))
                    skeleton_cc = max(skeleton_comp, key=len)
                    segment_subgraph = segment_subgraph.subgraph(skeleton_cc).copy()

                if skeleton_graph.degree[skeleton_node] != 1:
                    degree_1_nodes = [node for node in segment_subgraph.nodes() if skeleton_graph.degree[node] == 1]
                    if degree_1_nodes:
                        skeleton_node = degree_1_nodes[0]

                sorted_nodes = list(nx.dfs_preorder_nodes(segment_subgraph, skeleton_node))
                sorted_coords = np.array([segment_subgraph.nodes[n]['position'][:3] for n in sorted_nodes])
                sorted_radii = np.array([radii[n] for n in segment_indices])

                aligned_coords = sorted_coords - sorted_coords[0]
                rotated_coords = utils_process.rotate_curve(aligned_coords)
                rotated_coords = np.hstack((rotated_coords, sorted_radii.reshape(-1, 1)))

                length, distance, curvature, direction_vector = utils_process.calculate_segment_properties(
                    rotated_coords)
                label_radii = radii[np.where(labels == label)[0]]
                avg_radius = np.mean(label_radii) * 100 if len(label_radii) > 0 else 0
                condition = [avg_radius, length, distance, curvature]

                sample_points = utils_process.resample_curve(rotated_coords, target_distance=0.02)
                data = utils_process.calculate_distance(sample_points)
                segment_graph = nx.Graph()
                for idx, point in enumerate(data):
                    segment_graph.add_node(idx, position=point[:4])
                    if idx > 0:
                        segment_graph.add_edge(idx - 1, idx)
                utils.graph_to_ply(segment_graph, os.path.join(output_seq_path, f'{patient_id}_{key_node}.ply'))

                data = data.astype(np.float32)
                all_sequences.append(data)
                all_conditions.append(condition)
                all_lengths.append(data.shape[0])
                all_filenames.append(f"{patient_id}_{key_node}")

                key_graph.nodes[key_node]['position'] = np.array(
                    key_graph.nodes[key_node]['position'][:3] + condition + direction_vector.tolist())

            utils.graph_to_ply(key_graph, os.path.join(output_key_path, f'{patient_id}.ply'))

    max_len = min(max_seq_len, max(all_lengths))
    print(f"Max sequence length: {max(all_lengths)}")
    print(f"Final sequence length: {max_len}")
    padded_sequences = np.zeros((len(all_sequences), max_len, all_sequences[0].shape[-1]), dtype=np.float32)

    for i, seq in enumerate(all_sequences):
        seq_len = min(seq.shape[0], max_len)
        padded_sequences[i, :seq_len] = seq[:seq_len]

    output_data = {
        'sequences': torch.from_numpy(padded_sequences),
        'conditions': torch.tensor(all_conditions, dtype=torch.float32),
        'lengths': torch.tensor(all_lengths, dtype=torch.int32),
        'filenames': all_filenames}

    torch.save(output_data, os.path.join(output_path, f'{datasetname}_{max_len}.pt'))

    conditions_path = os.path.join(output_path, 'conditions.txt')
    np.savetxt(conditions_path, np.array(all_conditions), fmt='%.6f')
