import argparse


def seq_args():
    parser = argparse.ArgumentParser(description="Transformer VAE for Sequence Generation")
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--condition_dim', type=int, default=5)
    parser.add_argument('--max_seq_len', type=int, default=200)

    parser.add_argument('--data_path', type=str, default=r'./data/')
    # Model
    parser.add_argument('--input_dim', type=int, default=8)
    parser.add_argument('--latent_dim', default=64, type=int)

    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--best_avg_metric', type=int, default=100)
    # Training
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=2e-04, type=float)

    parser.add_argument('--test_interval', default=2000, type=int)
    parser.add_argument('--num_samples', default=300, type=int)
    parser.add_argument('--model_ckp', default=200, type=int)
    parser.add_argument('--current_epoch', default=0, type=int)
    parser.add_argument('--log_dir', type=str, default=r'./logs/')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--kl_weight', default=5, type=float)
    parser.add_argument('--len_weight', default=1.5, type=float)
    parser.add_argument('--recon_weight', default=1, type=float)

    args = parser.parse_args()
    return args


def tree_args():
    parser = argparse.ArgumentParser(description="TREE Model Arguments")
    parser.add_argument('--data_path', type=str, default=r'./data/')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--input_size', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--checkpoint', default=2000, type=int)

    parser.add_argument('--epochs', type=int, default=40000)

    parser.add_argument('--log_dir', type=str, default=r'./logs/')
    parser.add_argument('--output_dir', type=str, default=r'./output/')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--mmd_distance', type=str, default='rbf')
    parser.add_argument('--max_subgraph', type=bool, default=True)

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_step_size', default=100, type=float)
    parser.add_argument('--lr_gamma', default=1, type=float)
    parser.add_argument('--kl_weight', default=0.001, type=float)

    parser.add_argument('--latent_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2001)

    args = parser.parse_args()
    return args


