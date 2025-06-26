import torch
import os
import networkx as nx
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation
from model.model_seq import TransformerVAE
from model.model_tree import RecursiveDecoder

import networkx as nx


def remove_edge(graph):
    modified_graph = graph.copy()

    candidate_edges = []
    for u, v in modified_graph.edges():
        if modified_graph.degree[u] == 2 and modified_graph.degree[v] == 2:
            candidate_edges.append((u, v))

    if not candidate_edges:
        return modified_graph, None

    if len(candidate_edges) == 1:
        u, v = candidate_edges[0]
        if modified_graph.has_edge(u, v):
            modified_graph.remove_edge(u, v)
        return modified_graph, (u, v)

    best_edge = None
    best_diff = float('inf')

    for (u, v) in candidate_edges:
        if modified_graph.has_edge(u, v):
            modified_graph.remove_edge(u, v)

            components = list(nx.connected_components(modified_graph))
            if len(components) == 2:
                size1, size2 = len(components[0]), len(components[1])
                diff = abs(size1 - size2)
                if diff < best_diff:
                    best_diff = diff
                    best_edge = (u, v)

            modified_graph.add_edge(u, v)

    if best_edge is not None:
        u, v = best_edge
        if modified_graph.has_edge(u, v):
            modified_graph.remove_edge(u, v)

        return modified_graph, (u, v)

    return modified_graph, None


def unify_graph_positions(graph):
    for n in graph.nodes:
        pos = graph.nodes[n]['position']
        pos_1d = np.array(pos).flatten()
        graph.nodes[n]['position'] = pos_1d


def create_result_folder(args, mode):
    current_time = datetime.now().strftime('%m_%d_%H_%M')
    output_path = os.path.join(args.result_path, args.dataset + '_' + mode + '_' + current_time)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def load_model(tree_args, seq_args, assembly_args):
    if seq_args is not None:
        model_seq = TransformerVAE(input_dim=seq_args.input_dim, hidden_dim=seq_args.hidden_dim,
                                   latent_dim=seq_args.latent_dim,
                                   num_layers=seq_args.num_layers, nhead=seq_args.n_head,
                                   max_seq_len=assembly_args.max_seq_len,
                                   condition_dim=seq_args.condition_dim).to(seq_args.device)
        model_seq.load_state_dict(torch.load(os.path.join(assembly_args.seq_path, assembly_args.model_name)))
        # model_seq.load_state_dict(torch.load(os.path.join(assembly_args.seq_path, 'best_model.pth')))

        if tree_args is not None:
            model_tree = RecursiveDecoder(latent_size=tree_args.latent_size, hidden_size=tree_args.hidden_size,
                                          output_size=tree_args.input_size, args=tree_args).to(tree_args.device)
            # checkpoint = torch.load(os.path.join(assembly_args.tree_path, 'best_model.pth'))
            checkpoint = torch.load(os.path.join(assembly_args.tree_path, 'best_model.pth'))
            model_tree.load_state_dict(checkpoint['decoder'])
            return model_tree, model_seq
        else:
            return model_seq


def resample_curve(points, target_distance=0.01):
    filtered_points = [points[0]]
    for i in range(1, len(points) - 1):
        distance = np.linalg.norm(points[i] - filtered_points[-1])
        if distance >= target_distance:
            filtered_points.append(points[i])
    filtered_points.append(points[-1])
    filtered_points = np.array(filtered_points)

    distances = np.sqrt(np.sum(np.diff(filtered_points, axis=0) ** 2, axis=1))
    cumulative_length = np.cumsum(distances)
    cumulative_length = np.insert(cumulative_length, 0, 0)
    total_length = cumulative_length[-1]

    if target_distance >= total_length:
        if len(filtered_points) == 1:
            np.append(filtered_points, [points[-1]], axis=0)
        return filtered_points

    num_new_points = max(2, int(total_length / target_distance))
    new_sample_points = np.linspace(0, total_length, num_new_points)

    resampled_points = np.empty((num_new_points, filtered_points.shape[1]))
    for dim in range(filtered_points.shape[1]):
        resampled_points[:, dim] = np.interp(new_sample_points, cumulative_length, filtered_points[:, dim])

    return resampled_points


def rotate_curve_around_axis(points, rotation_axis, current_direction, target_direction):
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    current_direction = current_direction / np.linalg.norm(current_direction)
    target_direction = target_direction / np.linalg.norm(target_direction)

    cos_alpha = np.dot(current_direction, target_direction)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    cross_product = np.cross(current_direction, target_direction)
    dot_product = np.dot(cross_product, rotation_axis)

    if dot_product < 0:
        alpha = -alpha
    alpha += np.pi
    cos_theta = np.cos(alpha)
    sin_theta = np.sin(alpha)
    axis_x, axis_y, axis_z = rotation_axis

    rotation_matrix = np.array([
        [cos_theta + axis_x ** 2 * (1 - cos_theta),
         axis_x * axis_y * (1 - cos_theta) - axis_z * sin_theta,
         axis_x * axis_z * (1 - cos_theta) + axis_y * sin_theta],
        [axis_y * axis_x * (1 - cos_theta) + axis_z * sin_theta,
         cos_theta + axis_y ** 2 * (1 - cos_theta),
         axis_y * axis_z * (1 - cos_theta) - axis_x * sin_theta],
        [axis_z * axis_x * (1 - cos_theta) - axis_y * sin_theta,
         axis_z * axis_y * (1 - cos_theta) + axis_x * sin_theta,
         cos_theta + axis_z ** 2 * (1 - cos_theta)]
    ])

    p1 = points[0]
    translated_points = points - p1
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    rotated_points = rotated_points + p1

    return rotated_points, alpha


def compute_direction_vector(segment_coords):
    start_point = segment_coords[0]
    end_point = segment_coords[-1]
    direction_vector = end_point - start_point

    max_distance = 0
    max_point = None
    for point in segment_coords:
        distance = np.linalg.norm(np.cross(direction_vector, point - start_point)) / np.linalg.norm(direction_vector)
        if distance > max_distance:
            max_distance = distance
            max_point = point

    if max_point is not None:
        direction_vector = end_point - max_point

    return direction_vector


def transform_curve(curve, node1_data, node2_data, target_distance):
    radius = np.abs(curve[:, 3].cpu().numpy().reshape(-1, 1))
    radius = radius[::-1]
    curve = curve[:, :3].cpu().numpy().reshape(-1, 3)

    node1_position = np.array(node1_data.get('position'))[:3]
    node2_position = np.array(node2_data.get('position'))[:3]
    key_rotation_axis = node2_position - node1_position

    curve_start_point = curve[0]
    curve_end_point = curve[-1]

    original_length = np.linalg.norm(curve_end_point - curve_start_point)
    target_length = np.linalg.norm(node2_position - node1_position)
    scale_factor = target_length / original_length
    scaled_curve = curve_start_point + scale_factor * (curve - curve_start_point)

    curve_end_point_scaled = scaled_curve[-1]
    curve_start_point_scaled = scaled_curve[0]
    translation = node1_position - curve_start_point
    curve_rotation_axis = curve_end_point_scaled - curve_start_point_scaled

    rotation_result = Rotation.align_vectors([key_rotation_axis], [curve_rotation_axis])
    rotation_matrix_align = rotation_result[0].as_matrix()

    centered_points = scaled_curve - curve_start_point_scaled
    rotated_points = np.dot(centered_points, rotation_matrix_align.T)
    rotated_points = rotated_points + curve_start_point_scaled + translation
    rotated_points = np.hstack((rotated_points, radius))
    resampled_points = resample_curve(rotated_points, target_distance=target_distance)
    curve_direction_vector = compute_direction_vector(resampled_points[:, :3])
    key_direction_vector = np.array(node2_data.get('position'))[-3:]
    resampled_points[:, :3], alpha = rotate_curve_around_axis(resampled_points[:, :3], key_rotation_axis,
                                                              curve_direction_vector, key_direction_vector)

    curve_graph = nx.Graph()
    for i, point in enumerate(resampled_points):
        curve_graph.add_node(i, position=tuple(point))
    for i in range(len(resampled_points) - 1):
        curve_graph.add_edge(i, i + 1)

    return curve_graph


def find_edge_to_process(graph, dataset):
    edges_to_process = []
    root_node = None
    if dataset == 'imagecas':
        remove_graph, removed_edge = remove_edge(graph)
        if removed_edge is not None:
            (root_node, v) = removed_edge
            dfs_edges = nx.dfs_edges(graph, source=root_node)
            edges_to_process.extend((u, w) for u, w in dfs_edges if (u, w) != (root_node, v))
        else:
            root_node = 0
            edges_to_process.extend(nx.dfs_edges(graph, source=root_node))

    elif dataset == 'march':
        leaf_nodes = [node for node, degree in graph.degree() if degree == 1]
        root_node = min(leaf_nodes, key=lambda node: sum(np.array(graph.nodes[node]['position']) ** 2))
        edges_to_process.extend(nx.dfs_edges(graph, source=root_node))

    elif dataset == 'intra':
        leaf_nodes = [node for node, degree in graph.degree() if degree == 1]
        root_node = max(leaf_nodes, key=lambda node: graph.nodes[node]['position'][0])
        edges_to_process.extend(nx.dfs_edges(graph, source=root_node))

    elif dataset == 'cow':
        leaf_nodes = [node for node, degree in graph.degree() if degree == 1]
        root_node = min(leaf_nodes, key=lambda node: graph.nodes[node]['position'][2])
        edges_to_process.extend(nx.dfs_edges(graph, source=root_node))

    return edges_to_process, root_node


def traverse_and_paste_curves(graph, model, arg, dataset, target_distance):
    unify_graph_positions(graph)
    skeleton_graph = nx.Graph()
    key_node_to_skel_node = {}
    next_node_id = 0

    edges_to_process, root_node = find_edge_to_process(graph, dataset)
    depth_dict = nx.single_source_shortest_path_length(graph, root_node)
    min_depth = min(depth_dict.values())
    max_depth = max(depth_dict.values())

    for u, v in edges_to_process:
        depth_u = depth_dict.get(u, 0)
        depth_v = depth_dict.get(v, 0)
        if depth_u > depth_v:
            u, v = v, u
        node1_data = graph.nodes[u]
        node2_data = graph.nodes[v]
        depth = depth_dict.get(u, 0)
        normalized_depth = (depth - min_depth) / (max_depth - min_depth) if max_depth != min_depth else 0
        cond_array = np.array(node2_data['position'][3:-3])
        cond_array = np.append(cond_array, normalized_depth).reshape(1, -1)
        cond_tensor = torch.tensor(cond_array, device=arg.device, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            z = torch.randn((1, arg.latent_dim), device=arg.device)
            generated = model.decoder.sample(z, cond_tensor)

        curve_graph = transform_curve(generated, node1_data, node2_data, target_distance)

        curve_points = []
        curve_node_ids = list(range(curve_graph.number_of_nodes()))
        for i in curve_node_ids:
            curve_points.append(curve_graph.nodes[i]['position'])
        curve_points = np.array(curve_points)
        num_curve_pts = len(curve_points)

        if u not in key_node_to_skel_node:
            skel_node_u_id = next_node_id
            key_node_to_skel_node[u] = skel_node_u_id
            skeleton_graph.add_node(skel_node_u_id, position=tuple(curve_points[0]))
            next_node_id += 1
        else:
            skel_node_u_id = key_node_to_skel_node[u]

        if v not in key_node_to_skel_node:
            skel_node_v_id = next_node_id
            key_node_to_skel_node[v] = skel_node_v_id
            skeleton_graph.add_node(skel_node_v_id, position=tuple(curve_points[-1]))
            next_node_id += 1
        else:
            skel_node_v_id = key_node_to_skel_node[v]

        previous_node_id = skel_node_u_id
        for i in range(1, num_curve_pts - 1):
            current_node_id = next_node_id
            skeleton_graph.add_node(current_node_id, position=tuple(curve_points[i]))
            skeleton_graph.add_edge(previous_node_id, current_node_id)
            previous_node_id = current_node_id
            next_node_id += 1
        skeleton_graph.add_edge(previous_node_id, skel_node_v_id)

    return skeleton_graph


def smooth_bifurcation_node(graph, iterations=5, smooth_factor=0.5):
    bifurcation_nodes = [node for node, degree in graph.degree() if degree == 3]
    nodes_to_smooth = set(bifurcation_nodes)
    for node in bifurcation_nodes:
        neighbors_1 = list(graph.neighbors(node))
        nodes_to_smooth.update(neighbors_1)
        for neighbor in neighbors_1:
            neighbors_2 = list(graph.neighbors(neighbor))
            nodes_to_smooth.update(neighbors_2)
        for neighbor in neighbors_2:
            neighbors_3 = list(graph.neighbors(neighbor))
            nodes_to_smooth.update(neighbors_3)

    for _ in range(iterations):
        for node in nodes_to_smooth:
            node_pos = graph.nodes[node]['position']
            neighbor_positions = []
            for neighbor in graph.neighbors(node):
                neighbor_positions.append(graph.nodes[neighbor]['position'])
                for n2 in graph.neighbors(neighbor):
                    neighbor_positions.append(graph.nodes[n2]['position'])
                    for n3 in graph.neighbors(n2):
                        neighbor_positions.append(graph.nodes[n3]['position'])

            if neighbor_positions:
                mean_position = np.mean(neighbor_positions, axis=0)
                graph.nodes[node]['position'] += smooth_factor * (mean_position - node_pos)

    return graph
