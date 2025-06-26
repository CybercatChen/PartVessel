from model.model_tree import RecursiveEncoder, RecursiveDecoder
from utils.torch_f import Fold, encode_structure_fold, decode_structure_fold
from utils.utils import setup_logging, graph_to_ply
from test_tree import decode_testing
from evaluation.structure_evaluator import get_stats_eval

import numpy as np
import time
import logging


def train_one_epoch(encoder, decoder, dataloader, optimizer, arg):
    encoder.train()
    decoder.train()
    total_recon_loss = 0
    total_kl_loss = 0

    for _, (trees, num_nodes, graphs, file_names) in enumerate(dataloader):
        enc_fold = Fold(arg.device)
        enc_fold_nodes = [encode_structure_fold(enc_fold, tree) for tree in trees]
        enc_fold_nodes = enc_fold.apply(encoder, [enc_fold_nodes])
        enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)

        dec_fold = Fold(arg.device)
        dec_fold_nodes = []
        kld_fold_nodes = []

        for tree, fold_node in zip(trees, enc_fold_nodes):
            root_code, kl_div = torch.chunk(fold_node, 2, 1)
            dec_fold_nodes.append(decode_structure_fold(dec_fold, root_code, tree))
            kld_fold_nodes.append(kl_div)

        total_loss = dec_fold.apply(decoder, [dec_fold_nodes, kld_fold_nodes])
        num_nodes = torch.as_tensor(num_nodes, device=arg.device)
        recon_loss = torch.div(total_loss[0], num_nodes).sum() / len(trees)
        kl_loss = torch.stack(kld_fold_nodes).sum() / len(trees)
        loss = recon_loss + arg.kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    total_recon_loss /= len(dataloader)
    total_kl_loss /= len(dataloader)

    return total_recon_loss, total_kl_loss


def test(encoder, decoder, dataloader, arg, mode='Test'):
    encoder.eval()
    decoder.eval()
    all_results = []
    all_cd_results = []

    with torch.no_grad():
        for (trees, num_nodes, gt_graphs, file_names) in dataloader:
            test_enc_fold = Fold(arg.device)
            test_enc_fold_nodes = [encode_structure_fold(test_enc_fold, tree) for tree in trees]
            test_enc_fold_nodes = test_enc_fold.apply(encoder, [test_enc_fold_nodes])
            test_enc_fold_nodes = torch.split(test_enc_fold_nodes[0], 1, 0)

            recon_graphs = []
            for test_fold_node in test_enc_fold_nodes:
                test_root_code, _ = torch.chunk(test_fold_node, 2, 1)
                recon_tree = decode_testing(vector=test_root_code, max=50, decoder=decoder)
                recon_graph = recon_tree.to_graph(dec=True)
                recon_graphs.append(recon_graph)

            for i, (gt_graph, recon_graph) in enumerate(zip(gt_graphs, recon_graphs)):
                graph_to_ply(gt_graph, os.path.join(arg.log_dir, f'{file_names[i]}_gt.ply'))
                graph_to_ply(recon_graph, os.path.join(arg.log_dir, f'{file_names[i]}_pre.ply'))

            stats_eval_fn = get_stats_eval(arg)
            stats_results = stats_eval_fn(gt_graphs, recon_graphs)
            all_results.append(stats_results)

    eval_results = {key: np.mean([result[key] for result in all_results]) for key in all_results[0].keys()}
    logging.info(f"{mode} | " + " | ".join(
        [f"{key}: {value:.4f}" for key, value in eval_results.items()]
    ))

    return eval_results


def train_model(encoder, decoder, train_data, test_data, optimizer, arg):
    best_cd = 10
    writer, arg.log_dir = setup_logging(arg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg.lr_step_size, gamma=arg.lr_gamma)
    for epoch in range(arg.epochs):
        start_time = time.time()

        total_recon_loss, total_kl_loss = train_one_epoch(encoder, decoder, train_data, optimizer, arg)
        writer.add_scalar('Train/Recon_Loss', total_recon_loss, epoch)
        writer.add_scalar('Train/KL_Loss', total_kl_loss, epoch)

        if (epoch + 1) % arg.checkpoint == 0:
            eval_results_test = test(encoder, decoder, test_data, arg, mode='Test')
            eval_results_train = test(encoder, decoder, train_data, arg, mode='Train')

            for key, value in eval_results_test.items():
                writer.add_scalar(f'Test/{key}', value, epoch)
            for key, value in eval_results_train.items():
                writer.add_scalar(f'Train/{key}', value, epoch)

            epoch_cd = eval_results_test['chamfer_distance']
            if epoch_cd < best_cd:
                best_cd = epoch_cd
                torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                           os.path.join(arg.log_dir, 'models', f'best_model.pth'))
                logging.info(f'New best model saved with Chamfer Distance: {best_cd:.4f}')

            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                       os.path.join(arg.log_dir, 'models', f'{epoch}.pth'))
        scheduler.step()
        end_time = time.time()
        logging.info(f"Epoch [{epoch + 1}/{arg.epochs}] | Time: {end_time - start_time:.2f}s | "
                     f"Recon Loss: {total_recon_loss:.4f} | KL Loss: {total_kl_loss:.4f}")

    return encoder, decoder


if __name__ == '__main__':
    from config import args
    from torch.utils.data import DataLoader
    from utils.dataset import *
    from utils.utils import set_seed

    set_seed(args.seed)
    train_dataset = TreeDataset(args.dataset, args.data_path, is_train=True)
    test_dataset = TreeDataset(args.dataset, args.data_path, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True,
                                  collate_fn=coll_function)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True,
                                 collate_fn=coll_function)

    Encoder = RecursiveEncoder(input_size=args.input_size, feature_size=args.latent_size,
                               hidden_size=args.hidden_size).to(args.device)
    Decoder = RecursiveDecoder(latent_size=args.latent_size, hidden_size=args.hidden_size,
                               output_size=args.input_size, args=args).to(args.device)

    opt = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=args.lr)

    train_model(encoder=Encoder, decoder=Decoder, train_data=train_dataloader, test_data=test_dataloader,
                optimizer=opt, arg=args)
