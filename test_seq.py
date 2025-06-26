import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.dataset import Skeleton
from utils.utils import generate_masks, numpy_to_ply
from torch.utils.data import DataLoader


def inter_gen(model, z_start, z_end, num_steps=10):
    interpolated_samples = []
    for alpha in np.linspace(0, 1, num_steps):
        z_interpolated = z_start * (1 - alpha) + z_end * alpha
        generated_sample = model.decoder.sample(z_interpolated, condition=None)
        interpolated_samples.append(generated_sample.cpu().detach().numpy())
    return interpolated_samples


def visual_latent(model, data_loader, args, model_path):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for i, (data, condition) in enumerate(data_loader):
            pad_mask, tgt_mask = generate_masks(data)
            data = data.to(args.device, dtype=torch.float32)

            condition = condition.to(args.device, dtype=torch.float32)
            pad_mask = pad_mask.to(args.device, dtype=torch.bool)
            tgt_mask = tgt_mask.to(args.device, dtype=torch.bool)

            recon, length, mu, log_var = model(data, condition, pad_mask, tgt_mask)
            latent_vectors.append(mu.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)

    tsne = TSNE(n_components=2, perplexity=30.0)
    latent_tsne = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], cmap='viridis')
    plt.colorbar()
    plt.title("t-SNE Visualization of Latent Space")
    plt.savefig(os.path.join(model_path, 'test/latent_space_tsne.png'))


def condition_gen(model, conditions, arg, output_dir, num_samples=None, save_files=False):
    generation = []
    model.eval()

    with torch.no_grad():
        conditions_to_generate = conditions[:num_samples] if num_samples else conditions

        for idx, condition in enumerate(conditions_to_generate):
            condition = condition.reshape(1, -1)
            if isinstance(condition, np.ndarray):
                condition = torch.from_numpy(condition).float()
            condition = condition.to(arg.device)
            z = torch.randn((1, arg.latent_dim)).to(arg.device)
            generated_sample = model.decoder.sample(z, condition)
            generated_sample[:, 3] = generated_sample[:, 3] / 100
            generation.append(generated_sample.cpu().detach().numpy())

            if save_files:
                numpy_to_ply(generated_sample.cpu().detach().numpy(),
                             os.path.join(output_dir, f'sample_{idx}.ply'))

    return generation


def test(arg, model_path, epoch):
    test_dataset = Skeleton(arg.data_path, arg.dataset, max_seq_len=arg.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=0)

    model = TransformerVAE(input_dim=arg.input_dim, hidden_dim=64, latent_dim=64,
                           num_layers=4, nhead=4, max_seq_len=arg.max_seq_len).to(arg.device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model_epoch_{}.pth'.format(epoch))))

    os.makedirs(os.path.join(model_path, 'test'), exist_ok=True)
    visual_latent(model, test_loader, arg, model_path)

    z_start = torch.randn((1, arg.latent_dim)).to(arg.device)
    z_end = torch.randn((1, arg.latent_dim)).to(arg.device)
    interpolated_samples = inter_gen(model, z_start, z_end)

    for i, sample in enumerate(interpolated_samples):
        numpy_to_ply(sample, os.path.join(model_path, 'test/inter_{}.ply'.format(i)), mode=arg.dataset)

    condition = torch.tensor([[0.57, 0.34, 0.403]], device='cuda')
    condition_gen(model, condition, arg, model_path)


if __name__ == '__main__':
    from config import arg
    from model.model_seq import TransformerVAE
    from datetime import datetime

    model = TransformerVAE(input_dim=arg.input_dim, hidden_dim=64, latent_dim=64,
                           num_layers=4, nhead=4, max_seq_len=arg.max_seq_len, condition_dim=arg.condition_dim).to(
        arg.device)
    model_path = r'.\logs\cow_200_level_02_27_06_49'
    model.load_state_dict(torch.load(os.path.join(model_path, 'model_epoch_800.pth')))

    conditions = np.loadtxt(r'.\result\conditions_level_cow.txt')
    output_path = r'D:\PartVessel\TransVAE\result'
    current_time = datetime.now().strftime('%m_%d_%H_%M')
    log_dir = os.path.join(output_path, current_time)
    os.makedirs(log_dir, exist_ok=True)
    condition_gen(model=model, conditions=conditions, arg=arg, output_dir=log_dir, num_samples=10, save_files=True)
