import math
import torch.nn as nn
import networkx as nx
import torch


class StreetMoverDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction='none'):
        super(StreetMoverDistance, self).__init__()
        self.sinkhorn_distance = SinkhornDistance(eps=eps, max_iter=max_iter, reduction=reduction)

    def forward(self, y_A, y_nodes, output_A, output_nodes, n_points=100):
        y_pc = self.get_point_cloud(y_A, y_nodes, n_points)
        output_pc = self.get_point_cloud(output_A, output_nodes, n_points)
        sink_dist, P, C = self.sinkhorn_distance(y_pc, output_pc)
        return (y_pc, output_pc), (sink_dist, P, C)

    def get_point_cloud(self, A, nodes, n_points):
        n_divisions = n_points - 1 + 0.01
        total_len = get_cumulative_distance(A, nodes)
        step = total_len / n_divisions if total_len > 0 else 0
        points = []
        next_step = 0.
        used_len = 0.

        for i in range(A.shape[0]):
            for j in range(i):
                if A[i, j] == 1.:
                    next_step, used, pts = get_points(next_step, step, nodes[j].clone(), nodes[i].clone())
                    used_len += used
                    points += pts
                    last_node = nodes[i].clone()
        if len(points) < n_points:
            fill_point = (last_node.tolist() if last_node is not None else [0.0, 0.0, 0.0])
            while len(points) < n_points:
                points.append(tuple(fill_point))
        if len(points) == 0:
            return torch.zeros((n_points, 3))
        return torch.FloatTensor(points)


def get_cumulative_distance(A, nodes):
    tot = 0.
    for i in range(A.shape[0]):
        for j in range(i):
            if A[i, j] == 1.:
                tot += euclidean_distance(nodes[i], nodes[j])
    return tot


def get_points(next_step, step, a, b):
    l = euclidean_distance(a, b)
    direction = (b - a) / l if l > 0 else torch.zeros_like(a)
    pts = []
    used = 0.
    while next_step <= l:
        used += next_step
        l -= next_step
        a = a + direction * next_step
        pts.append((a[0].item(), a[1].item(), a[2].item()))
        next_step = step
    next_step = step - l
    return next_step, used, pts


def euclidean_distance(a, b):
    return math.sqrt((a - b).pow(2).sum().item())


class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        C = self._cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = torch.empty(batch_size, x_points, dtype=torch.float).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        actual_nits = 0
        thresh = 1e-1

        for i in range(self.max_iter):
            u1 = u.clone()
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        return tau * u + (1 - tau) * u1


def graph_to_data(graph):
    nodes_sorted = sorted(graph.nodes())
    A = nx.to_numpy_array(graph, nodelist=nodes_sorted)
    coords = [graph.nodes[node]['position'][:3] for node in nodes_sorted]
    A_tensor = torch.FloatTensor(A)
    coords_tensor = torch.FloatTensor(coords)
    return A_tensor, coords_tensor


def compute_streetmover_distance(gt_graph, pred_graph, n_points=500):
    A_gt, nodes_gt = graph_to_data(gt_graph)
    A_pred, nodes_pred = graph_to_data(pred_graph)
    mover_distance = StreetMoverDistance(eps=0.00001, max_iter=10)
    (point_cloud_gt, point_cloud_pred), (sink_dist, P, C) = mover_distance(
        A_gt, nodes_gt, A_pred, nodes_pred, n_points=n_points
    )

    return sink_dist
