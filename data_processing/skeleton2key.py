import numpy as np
import os
import networkx as nx
import utils_process
from utils import utils


def get_key_graph(key_graph_folder, skeleton_folder, label_folder, radius_folder, output_path, dataset='imagecas'):
    conditions = []
    os.makedirs(os.path.join(output_path, 'key_graph'), exist_ok=True)
    if dataset == 'imagecas':
        os.makedirs(os.path.join(output_path, 'heart_graph'), exist_ok=True)
    for skeleton_file in os.listdir(skeleton_folder):
        if skeleton_file.endswith('.ply'):
            patient_id = skeleton_file.split('.')[0]
            skeleton_path = os.path.join(skeleton_folder, skeleton_file)
            label_path = os.path.join(label_folder, patient_id + '.ply.skeleton.seg')
            key_graph_path = os.path.join(key_graph_folder, patient_id + '.ply')
            radius_path = os.path.join(radius_folder, patient_id + '.ply.radius')

            skeleton_graph = utils.ply_to_graph(skeleton_path)
            key_graph = utils.ply_to_graph(key_graph_path)
            labels = np.loadtxt(label_path)
            radii = np.loadtxt(radius_path)
            for key_node in key_graph.nodes():
                key_node_position = key_graph.nodes[key_node]['position'][:3]
                skeleton_node = utils_process.find_skeleton_node_by_position(skeleton_graph, key_node_position)

                label = labels[skeleton_node]
                segment_indices = np.where(labels == label)[0]
                segment_subgraph = skeleton_graph.subgraph(segment_indices)

                sorted_nodes = list(nx.dfs_preorder_nodes(segment_subgraph, skeleton_node))
                sorted_coords = np.array([list(segment_subgraph.nodes[node]['position'][:3]) for node in sorted_nodes])

                (curve_length, straight_distance, curvature,
                 normalized_direction) = utils_process.calculate_segment_properties(sorted_coords)
                if len(list(key_graph.neighbors(key_node))) == 0 and straight_distance < 0.25:
                    key_graph.remove_node(key_node)
                    continue
                label_radius_indices = np.where(labels == label)[0]
                label_radii = radii[label_radius_indices]
                average_radius = np.mean(label_radii) if len(label_radii) > 0 else 0

                key_graph.nodes[key_node]['position'] = np.array(
                    [*key_graph.nodes[key_node]['position'][:3], average_radius, curve_length, straight_distance,
                     curvature, *normalized_direction])
                condition = [average_radius, curve_length, straight_distance, curvature]
                conditions.append(condition)

            utils.graph_to_ply(key_graph, os.path.join(output_path, 'key_graph', patient_id + '.ply'))
            if dataset == 'imagecas':
                heart_graph = utils_process.connect_root_nodes(key_graph)
                utils.graph_to_ply(heart_graph, os.path.join(output_path, 'heart_graph', patient_id + '.ply'))

    np.savetxt(os.path.join(output_path, 'key_conditions.txt'), conditions, fmt='%f')
