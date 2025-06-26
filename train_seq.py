import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from utils.dataset import Skeleton
from evaluation.point_metric import evaluate_generation_condition, compute_l_d_c, evaluate_reconstruction
from model.model_seq import TransformerVAE
from test_seq import condition_gen
import utils.utils as utils


def train(model, train_loader, test_loader, optimizer, args, writer):
    metrics = {
        'CD': 0, 'EMD': 0, 'MSE': 0,
        'MMD_lengths': 0, 'MMD_distances': 0, 'MMD_curvatures': 0,
        'JSD_lengths': 0, 'JSD_distances': 0, 'JSD_curvatures': 0
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        args.current_epoch = epoch
        epoch_total_loss, epoch_recon_loss, epoch_kl_loss, epoch_length_loss, epoch_radius_loss = 0, 0, 0, 0, 0
        start_time = time.time()
        for i, (data, condition, filename) in enumerate(train_loader):
            pad_mask, tgt_mask = utils.generate_masks(data, args)
            data = data.to(args.device, dtype=torch.float32)
            condition = condition.to(args.device, dtype=torch.float32)
            pad_mask = pad_mask.to(args.device, dtype=torch.float32)
            tgt_mask = tgt_mask.to(args.device, dtype=torch.float32)
            true_radius = data[:, :, 3].to(args.device)

            recon, length_logit, radius, mu, log_var = model(data, condition, pad_mask, tgt_mask)
            recon_loss, kl_loss, length_loss, radius_loss = model.get_loss(recon, data, length_logit, mu, log_var,
                                                                           pad_mask, true_radius)
            total_loss = (args.recon_weight * recon_loss + args.kl_weight * kl_loss +
                          length_loss * args.len_weight + radius_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_length_loss += length_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_radius_loss += radius_loss.item()

        num_batches = len(train_loader)
        epoch_total_loss /= num_batches
        epoch_recon_loss /= num_batches
        epoch_length_loss /= num_batches
        epoch_kl_loss /= num_batches
        epoch_radius_loss /= num_batches

        epoch_time = time.time() - start_time
        logging.info(
            f"Epoch [{epoch}/{args.epochs}], loss: {epoch_total_loss:.4f}, "
            f"recon_loss: {epoch_recon_loss:.4f}, len_loss: {epoch_length_loss:.4f}, "
            f"kl_loss: {epoch_kl_loss:.4f}, radius_loss: {epoch_radius_loss:.4f}, "
            f"time: {epoch_time:.2f}s"
        )
        writer.add_scalar('Loss/total_loss', epoch_total_loss, epoch)
        writer.add_scalar('Loss/recon_loss', epoch_recon_loss, epoch)
        writer.add_scalar('Loss/length_loss', epoch_length_loss, epoch)
        writer.add_scalar('Loss/kl_loss', epoch_kl_loss, epoch)
        writer.add_scalar('Loss/radius_loss', epoch_radius_loss, epoch)

        if epoch % args.model_ckp == 0 or epoch == args.epochs:
            model_save_path = os.path.join(args.log_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)

        if epoch % args.test_interval == 0:
            metrics, best_avg_metric = test(model, test_loader, args, writer, metrics, args.best_avg_metric)
            args.best_avg_metric = best_avg_metric


def test(model, test_loader, args, writer, metrics, best_avg_metric):
    model.eval()
    recon_keys = ['CD', 'EMD', 'MSE']
    evaluate_keys = ['MMD_lengths', 'MMD_distances', 'MMD_curvatures',
                     'JSD_lengths', 'JSD_distances', 'JSD_curvatures']

    with torch.no_grad():
        for batch, (test_data, test_condition, filename) in enumerate(test_loader):
            test_pad_mask, test_tgt_mask = utils.generate_masks(test_data, args)
            test_data = test_data.to(args.device, dtype=torch.float32)
            test_condition = test_condition.to(args.device, dtype=torch.float32)
            test_pad_mask = test_pad_mask.to(args.device, dtype=torch.float32)
            test_tgt_mask = test_tgt_mask.to(args.device, dtype=torch.float32)

            test_recon, _, _, _, _ = model(test_data, test_condition, test_pad_mask,
                                           test_tgt_mask)
            generation = condition_gen(model, test_condition, args, args.log_dir,
                                       num_samples=args.num_samples, save_files=False)
            recon_result = evaluate_reconstruction(test_data, test_recon)
            generated_condition = compute_l_d_c(generation)
            evaluate_results = evaluate_generation_condition(
                test_condition[:generated_condition.shape[0], :], generated_condition
            )

            if batch % 5 == 0:
                for i, file in enumerate(filename[:args.num_samples]):
                    utils.numpy_to_ply(test_recon[i].cpu().detach().numpy(), args.log_dir + f'/{file}_recon.ply')
                    utils.numpy_to_ply(test_data[i].cpu().detach().numpy(), args.log_dir + f'/{file}_refer.ply')
                    utils.numpy_to_ply(generation[i], args.log_dir + f'/gen_{i}.ply')

            batch_results = {key: recon_result[key] for key in recon_keys}
            batch_results.update({key: evaluate_results[key] for key in evaluate_keys})

            metrics = utils.update_metrics(metrics, batch_results, len(test_loader))

        args.best_avg_metric = sum(metrics.values()) / len(metrics)
        if args.best_avg_metric < best_avg_metric:
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
            logging.info(f"Best model at epoch {args.current_epoch} with curvature {args.best_avg_metric:.4f}")

    for metric, avg_value in metrics.items():
        logging.info(f"Test Results - {metric}: {avg_value:.4f}")
        writer.add_scalar(f"eval/{metric}", avg_value, args.epochs)

    return metrics, args.best_avg_metric


def main(args):
    train_dataset = Skeleton(data_path=args.data_path, dataset_name=args.dataset, max_seq_len=args.max_seq_len,
                             is_train=True)
    test_dataset = Skeleton(data_path=args.data_path, dataset_name=args.dataset, max_seq_len=args.max_seq_len,
                            is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TransformerVAE(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                           num_layers=args.num_layers, nhead=args.n_head, max_seq_len=args.max_seq_len,
                           condition_dim=args.condition_dim).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    utils.set_seed(args.seed)

    writer, args.log_dir = utils.setup_logging(args)
    train(model, train_loader, test_loader, optimizer, args, writer)


if __name__ == '__main__':
    from config import tree_args

    tree_args = tree_args()
    main(args=tree_args)
    print('Finished Training')
