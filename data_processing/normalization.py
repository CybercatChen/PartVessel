import os
import trimesh
import numpy as np
from utils import utils


def normalize_mesh(mesh):
    length = max(mesh.bounding_box.extents)
    center = mesh.bounding_box.centroid
    scale = 1.0 / length

    mesh.vertices -= center
    mesh.vertices *= scale
    return mesh


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
    scale = 1.0 / length

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


def normalizes_graph(skeleton_graph):
    skel_coords = []
    for node in skeleton_graph.nodes():
        skel_coords.append(skeleton_graph.nodes[node]['position'][:3])
    skel_coords = np.array(skel_coords)

    min_coords = np.min(skel_coords, axis=0)
    max_coords = np.max(skel_coords, axis=0)
    center = (min_coords + max_coords) / 2.0
    extents = max_coords - min_coords
    length = np.max(extents)
    scale = 1.0 / length

    def normalize_coord(pos):
        return (np.array(pos[:3]) - center) * scale

    for node in skeleton_graph.nodes():
        orig_pos = skeleton_graph.nodes[node]['position']
        normalized_pos = normalize_coord(orig_pos)
        skeleton_graph.nodes[node]['position'][:3] = normalized_pos

    return skeleton_graph


def normalize_meshes_in_folder(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.obj') or file_name.endswith('.ply'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            mesh = trimesh.load(input_file_path, process=False, maintain_order=True)

            normalized_mesh = normalize_mesh(mesh)
            normalized_mesh.export(output_file_path)
            print(f"Normalized and saved {file_name} to {output_file_path}")


def normalize_skeleton_in_folder(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.obj') or file_name.endswith('.ply'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            graph = utils.ply_to_graph(input_file_path)

            graph = normalize_graph(graph)
            utils.graph_to_ply(graph, output_file_path)

            print(f"Normalized and saved {file_name} to {output_file_path}")


def normalize_graph(skeleton_graph):

    def normalize_coord(pos):
        return (np.array(pos[:3]) / 200) - 0.5

    for node in skeleton_graph.nodes():
        orig_pos = skeleton_graph.nodes[node]['position']
        normalized_pos = normalize_coord(orig_pos)
        skeleton_graph.nodes[node]['position'][:3] = normalized_pos

    return skeleton_graph
