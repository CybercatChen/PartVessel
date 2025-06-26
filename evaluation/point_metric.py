import numpy as np
import torch
import os
import trimesh
import utils.utils as utils
from evaluation.PyTorchEMD.emd import earth_mover_distance
from evaluation.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from evaluation.evaluation_metrics_3d import jsd_between_point_cloud_sets, compute_all_metrics, emd_approx


def normalize_point_cloud(pc):
    pc_centered = pc - np.mean(pc, axis=0)
    max_dist = np.max(np.linalg.norm(pc_centered, axis=1))
    if max_dist > 0:
        pc_normalized = pc_centered / max_dist
    else:
        pc_normalized = pc_centered
    return pc_normalized


def evaluate_reconstruction(ref_pcd, recon_pcd):
    ref_pcd = torch.tensor(ref_pcd[:, :, :3]).to(torch.float)
    recon_pcd = torch.tensor(recon_pcd[:, :, :3]).to(torch.float)
    recon_pcd = recon_pcd.to('cuda')
    ref_pcd = ref_pcd.to('cuda')
    # recon_pcd.requires_grad = True
    # ref_pcd.requires_grad = True
    # emd = earth_mover_distance(recon_pcd, ref_pcd, transpose=False).mean()
    emd = emd_approx(recon_pcd, ref_pcd).mean()
    chamfer = chamfer_3DDist()
    dist1, dist2, _, _ = chamfer(recon_pcd, ref_pcd)
    cd = (torch.mean(dist1) + torch.mean(dist2)) / 2
    return {
        "CD": cd,
        "EMD": emd
    }


def evaluate_generation(sample_pcs, ref_pcs, batch_size=32):
    jsd_result = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)
    sample_pcs = torch.tensor(sample_pcs).to('cuda')
    ref_pcs = torch.tensor(ref_pcs).to('cuda')
    pairwise_results = compute_all_metrics(sample_pcs, ref_pcs, batch_size)
    results = {
        'JSD': jsd_result,
    }
    results.update(pairwise_results)

    return pairwise_results


def sample_point_cloud(pc, num_points=2048):
    N = pc.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)
    return pc[idx]


def read_ply_folder(folder):
    points_list = []

    for recon_file in os.listdir(folder):
        recon_file = os.path.join(folder, recon_file)
        recon_pc = utils.read_pcd(recon_file)
        sampled_pc = sample_point_cloud(recon_pc, num_points=2048)
        normalized_pc = normalize_point_cloud(sampled_pc)
        points_list.append(normalized_pc)

    points = np.stack(points_list, axis=0)[:, :, :3]
    return points


def read_mesh_folder(folder, num_points=2048):
    points_list = []

    for mesh_file in os.listdir(folder):
        file_path = os.path.join(folder, mesh_file)
        mesh = trimesh.load(file_path)
        sampled_points, _ = trimesh.sample.sample_surface(mesh, num_points)
        normalized_pc = normalize_point_cloud(sampled_points)
        points_list.append(normalized_pc)

    points = np.stack(points_list, axis=0)
    return points
