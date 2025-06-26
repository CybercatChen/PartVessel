from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from utils.node import *
from utils import utils
from datetime import datetime
from utils.torch_f import Fold, encode_structure_fold
import torch


def encode_testing(root, encoder):
    def encode_node(node, encoder):
        if node.right is None and node.left is None:
            return encoder.leafEncoder(node.radius.reshape(-1, 10))
        elif node.left is None and node.right is not None:
            right_feature = encode_node(node.right, encoder)
            return encoder.internalEncoder(node.radius.reshape(-1, 10), right_feature)
        elif node.right is None and node.left is not None:
            left_feature = encode_node(node.left, encoder)
            return encoder.internalEncoder(node.radius.reshape(-1, 10), left_feature)
        else:
            right_feature = encode_node(node.right, encoder)
            left_feature = encode_node(node.left, encoder)
            return encoder.bifurcationEncoder(node.radius.reshape(-1, 10), right_feature, left_feature)

    root_feature = encode_node(root, encoder)
    z = encoder.sampleEncoder(root_feature)

    return z


def decode_testing(vector, max, decoder):
    def decode_node(vector, max, decoder):

        cl = decoder.nodeClassifier(vector)
        _, label = torch.max(cl, 1)
        label = label.data

        if label.item() == 0 and create_node.count <= max:
            node = decoder.featureDecoder(vector)
            return create_node(create_node.count, node)

        elif label.item() == 1 and create_node.count <= max:
            right, node = decoder.internalDecoder(vector)
            d = create_node(create_node.count, node)
            d.right = decode_node(right, max, decoder)
            return d

        elif label.item() == 2 and create_node.count <= max:
            left, right, node = decoder.bifurcationDecoder(vector)
            d = create_node(create_node.count, node)
            d.right = decode_node(right, max, decoder)
            d.left = decode_node(left, max, decoder)
            return d

    create_node.count = 0
    vector = decoder.sample_decoder(vector)
    dec = decode_node(vector, max, decoder)

    return dec


def visualize_latent_space(data_loader, encoder, perplexity=50, iterations=1000):
    latent_vectors = []
    num_nodes = []
    encoder.eval()
    for _, (trees, nodes, graphs, file_names) in enumerate(data_loader):
        for tree in trees:
            z = encode_testing(tree, encoder)
            latent_vectors.append(z.detach().cpu().numpy())
        num_nodes.append(list(nodes))

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    num_nodes = np.concatenate(num_nodes, axis=0)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=iterations, random_state=2001)
    tsne_results = tsne.fit_transform(latent_vectors)

    plt.figure()
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=num_nodes, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Latent Space')
    plt.show()


def generate_samples(decoder, arg, num_samples=100, max_depth=50):
    decoder.eval()
    current_time = datetime.now().strftime('%m_%d_%H_%M')
    recon_dir = os.path.join(arg.output_dir, arg.dataset + '_' + current_time, "generation")
    os.makedirs(recon_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(num_samples):
            random_latent = torch.randn(1, arg.latent_size, device=arg.device)

            new_tree = decode_testing(vector=random_latent, max=max_depth, decoder=decoder)
            new_graph = new_tree.to_graph(dec=True)

            utils.graph_to_ply(new_graph, os.path.join(recon_dir, f'generated_sample_{i}.ply'))


def reconstruction(encoder, decoder, dataloader, arg, max_depth, mode):
    decoder.eval()
    encoder.eval()
    current_time = datetime.now().strftime('%m_%d_%H_%M')
    recon_dir = os.path.join(arg.output_dir, arg.dataset + '_' + current_time, f"recon_{mode}")
    ref_dir = os.path.join(arg.output_dir, arg.dataset + '_' + current_time, f"ref_{mode}")
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (trees, num_nodes, gt_graphs, file_names) in enumerate(dataloader):
            recon_graphs = []
            for i, (tree, gt_graph) in enumerate(zip(trees, gt_graphs)):
                test_enc_fold = Fold(arg.device)
                test_enc_fold_nodes = [encode_structure_fold(test_enc_fold, tree) for tree in trees]
                test_enc_fold_nodes = test_enc_fold.apply(encoder, [test_enc_fold_nodes])
                test_enc_fold_nodes = torch.split(test_enc_fold_nodes[0], 1, 0)

                for test_fold_node in test_enc_fold_nodes:
                    test_root_code, _ = torch.chunk(test_fold_node, 2, 1)
                    recon_tree = decode_testing(vector=test_root_code, max=max_depth, decoder=decoder)
                    recon_graph = recon_tree.to_graph(dec=True)
                    recon_graphs.append(recon_graph)
                    base_name = os.path.splitext(file_names[i])[0]
                    utils.graph_to_ply(gt_graph, os.path.join(ref_dir, f"{base_name}_original.ply"))
                    utils.graph_to_ply(recon_graph, os.path.join(recon_dir, f"{base_name}_reconstructed.ply"))


if __name__ == '__main__':
    from config import tree_args
    from torch.utils.data import DataLoader
    from utils.dataset import *
    from model.model_tree import RecursiveEncoder, RecursiveDecoder

    tree_args = tree_args()
    tree_args.dataset = 'imagecas'
    tree_args.input_size = 10
    train_dataset = TreeDataset(tree_args.dataset, tree_args.data_path, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=tree_args.batch_size, num_workers=0, shuffle=True,
                                  collate_fn=coll_function)

    test_dataset = TreeDataset(tree_args.dataset, tree_args.data_path, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=tree_args.batch_size, num_workers=0, shuffle=True,
                                 collate_fn=coll_function)

    encoder = RecursiveEncoder(input_size=tree_args.input_size, feature_size=tree_args.latent_size,
                               hidden_size=tree_args.hidden_size).to(tree_args.device)

    decoder = RecursiveDecoder(latent_size=tree_args.latent_size, hidden_size=tree_args.hidden_size,
                               output_size=tree_args.input_size, args=tree_args).to(tree_args.device)
    check_point_path = r"./logs/imagecas_02_14_12_12/models/19999.pth"
    check_point = torch.load(check_point_path)
    encoder.load_state_dict(check_point['encoder'])
    decoder.load_state_dict(check_point['decoder'])

    generate_samples(decoder, tree_args, num_samples=100, max_depth=50)
    reconstruction(encoder, decoder, test_dataloader, tree_args, max_depth=50, mode='test')
    reconstruction(encoder, decoder, train_dataloader, tree_args, max_depth=50, mode='train')
