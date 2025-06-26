import numpy as np
import networkx as nx


def rotate_curve(curve, target_axis=np.array([1, 1, 1])):
    aligned_curve = curve - curve[0]

    start_point = aligned_curve[0]
    end_point = aligned_curve[-1]
    original_direction = end_point - start_point
    original_norm = np.linalg.norm(original_direction)

    original_direction = original_direction / original_norm
    target_axis = np.array(target_axis, dtype=np.float32)
    target_axis = target_axis / np.linalg.norm(target_axis)

    rotation_axis = np.cross(original_direction, target_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-6:
        if np.dot(original_direction, target_axis) < 0:
            return -aligned_curve
        else:
            return aligned_curve

    theta = np.arccos(np.clip(np.dot(original_direction, target_axis), -1.0, 1.0))
    k = rotation_axis / rotation_axis_norm
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    rotated_curve = aligned_curve @ R.T

    final_direction = rotated_curve[-1] - rotated_curve[0]
    final_direction /= np.linalg.norm(final_direction)

    return rotated_curve


def resample_curve(points, target_distance=0.01):
    filtered_points = [points[0]]
    for i in range(1, len(points)):
        distance = np.linalg.norm(points[i] - filtered_points[-1])
        if distance >= target_distance:
            filtered_points.append(points[i])

    filtered_points = np.array(filtered_points)

    distances = np.sqrt(np.sum(np.diff(filtered_points, axis=0) ** 2, axis=1))
    cumulative_length = np.cumsum(distances)
    cumulative_length = np.insert(cumulative_length, 0, 0)
    total_length = cumulative_length[-1]

    if target_distance >= total_length:
        return filtered_points

    num_new_points = max(2, int(total_length / target_distance))
    new_sample_points = np.linspace(0, total_length, num_new_points)

    resampled_points = np.empty((num_new_points, filtered_points.shape[1]))
    for dim in range(filtered_points.shape[1]):
        resampled_points[:, dim] = np.interp(new_sample_points, cumulative_length, filtered_points[:, dim])

    return resampled_points


def calculate_curve_metrics(points):
    distance = np.linalg.norm(points[-1] - points[0])

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    length = np.sum(segment_lengths)
    curve = 1 - (distance / length) if length != 0 else 0

    return length, distance, curve


def calculate_distance(points):
    diff = points[1:] - points[:-1]
    euclidean_distance = np.linalg.norm(diff, axis=1)
    euclidean_distance = np.hstack(([0], euclidean_distance))

    x_diff = np.hstack(([0], diff[:, 0]))
    y_diff = np.hstack(([0], diff[:, 1]))
    z_diff = np.hstack(([0], diff[:, 2]))

    result = np.column_stack((
        points,
        euclidean_distance[:, np.newaxis],
        x_diff[:, np.newaxis],
        y_diff[:, np.newaxis],
        z_diff[:, np.newaxis]
    ))
    return result


def connect_root_nodes(key_graph):
    connected_components = list(nx.connected_components(key_graph))

    if len(connected_components) == 1:
        return key_graph
    elif len(connected_components) > 2:
        sorted_components = sorted(connected_components, key=len, reverse=True)
        component_1 = sorted_components[0]
        component_2 = sorted_components[1]
    else:
        component_1 = connected_components[0]
        component_2 = connected_components[1]

    leaf_nodes_1 = [node for node in component_1 if key_graph.degree(node) == 1]
    if not leaf_nodes_1:
        root_1 = next(iter(component_1))
    else:
        root_1 = max(leaf_nodes_1, key=lambda node: key_graph.nodes[node]['position'][0])

    leaf_nodes_2 = [node for node in component_2 if key_graph.degree(node) == 1]
    if not leaf_nodes_2:
        root_2 = next(iter(component_2))
    else:
        root_2 = max(leaf_nodes_2, key=lambda node: key_graph.nodes[node]['position'][0])

    key_graph.add_edge(root_1, root_2)

    return key_graph


def compute_direction_vector(segment_coords):
    start_point = segment_coords[0]
    end_point = segment_coords[-1]
    axis_vector = end_point - start_point
    max_distance = 0
    max_point = segment_coords[1]
    for point in segment_coords:
        distance = np.linalg.norm(np.cross(axis_vector, point - start_point)) / np.linalg.norm(axis_vector)
        if distance > max_distance:
            max_distance = distance
            max_point = point

    direction_vector = max_point - start_point
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    return direction_vector


def calculate_segment_properties(segment_coords):
    segment_coords = np.array(segment_coords[:, :3])
    start_coord = segment_coords[0]
    end_coord = segment_coords[-1]
    straight_distance = np.linalg.norm(end_coord - start_coord)

    segment_vectors = np.diff(segment_coords, axis=0)
    curve_length = np.sum(np.linalg.norm(segment_vectors, axis=1))

    curvature = 1 - (straight_distance / curve_length) if curve_length != 0 else 0

    axis_vector = end_coord - start_coord
    max_distance = 0
    max_point = segment_coords[1]
    for point in segment_coords:
        distance = np.linalg.norm(np.cross(axis_vector, point - start_coord)) / np.linalg.norm(axis_vector)
        if distance > max_distance:
            max_distance = distance
            max_point = point

    direction_vector = max_point - start_coord
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    return curve_length, straight_distance, curvature, direction_vector


def find_skeleton_node_by_position(skeleton_graph, key_node_position):
    min_distance = float('inf')
    closest_node = None
    for node in skeleton_graph.nodes:
        pos = np.array(skeleton_graph.nodes[node]['position'])[:3]
        distance = np.linalg.norm(pos - key_node_position)
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    return closest_node


def normalize_graphs(skeleton_graph, key_graph):
    skel_coords = []
    for node in skeleton_graph.nodes():
        skel_coords.append(skeleton_graph.nodes[node]['position'][:3])
    skel_coords = np.array(skel_coords)

    min_coords = np.min(skel_coords, axis=0)
    max_coords = np.max(skel_coords, axis=0)
    center = (min_coords + max_coords) / 2.0
    extents = max_coords - min_coords
    length = np.max(extents)
    scale = 2.0 / length

    def normalize_coord(pos):
        return (np.array(pos[:3]) - center) * scale

    for node in skeleton_graph.nodes():
        orig_pos = skeleton_graph.nodes[node]['position']
        normalized_pos = normalize_coord(orig_pos)
        skeleton_graph.nodes[node]['position'][:3] = normalized_pos

    for node in key_graph.nodes():
        orig_pos = key_graph.nodes[node]['position']
        normalized_pos = normalize_coord(orig_pos)
        key_graph.nodes[node]['position'][:3] = normalized_pos

    return skeleton_graph, key_graph
