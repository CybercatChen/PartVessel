import random
import networkx as nx
from evaluation.graph_mmd import mmd_eval


def resample_graph(graph, target_node_count):
    nodes = list(graph.nodes())
    current_node_count = len(nodes)

    if current_node_count > target_node_count:
        nodes_to_remove = random.sample(nodes, current_node_count - target_node_count)
        graph.remove_nodes_from(nodes_to_remove)

    return graph


def get_stats_eval(arg):
    if arg.mmd_distance.lower() == 'rbf':
        method = [('degree', 1., 'argmax'),
                  ('spectral', 1., 'argmax'),
                  ('cluster', 1., 'argmax'), ]
    else:
        raise ValueError

    def eval_stats_fn(test_graphs, pred_graphs):
        sub_pred_G = []
        for G in pred_graphs:
            CGs = [G.subgraph(c) for c in nx.connected_components(G)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            sub_pred_G += [CGs[0]]
        pred_graphs = sub_pred_G

        results = mmd_eval(test_graphs, pred_graphs, method)
        return results

    return eval_stats_fn
