import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, nhead, max_seq_len, condition_dim):
        super(Encoder, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim * 2, nhead, hidden_dim * 4, dropout=0.1,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc_mu = nn.Linear(hidden_dim * 3, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 3, latent_dim)
        self.max_seq_len = max_seq_len

    def forward(self, sequence, condition, pad_mask):
        sequence = self.input_proj(sequence)
        sequence = self.pos_encoder(sequence)

        condition_embedding = self.condition_proj(condition).unsqueeze(1)
        condition_expanded = condition_embedding.expand(-1, self.max_seq_len, -1)

        sequence_with_condition = torch.cat([sequence, condition_expanded], dim=-1)
        memory = self.transformer_encoder(sequence_with_condition, src_key_padding_mask=pad_mask)

        memory_con = torch.cat([memory, condition_expanded], dim=-1)
        mu = self.fc_mu(memory_con.mean(dim=1))
        log_var = self.fc_var(memory_con.mean(dim=1))

        return mu, log_var


def auto_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, nhead, max_seq_len, condition_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.latent_to_hidden = nn.Linear(latent_dim * 2, hidden_dim * 2)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.condition_proj = nn.Linear(condition_dim, latent_dim)

        self.output_proj = nn.Linear(hidden_dim * 2, input_dim)
        self.output_proj_radius = nn.Linear(hidden_dim * 2, 1)

        self.condition_proj_1 = nn.Linear(3, hidden_dim)
        self.condition_proj_2 = nn.Linear(2, latent_dim)
        self.length_predictor = nn.Linear(latent_dim * 2, max_seq_len)

        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)

        decoder_layers = nn.TransformerDecoderLayer(hidden_dim * 2, nhead, hidden_dim * 4, dropout=0.1,
                                                    batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

        decoder_layers_radius = nn.TransformerDecoderLayer(hidden_dim * 2, nhead, hidden_dim * 4, dropout=0.1,
                                                           batch_first=True)
        self.transformer_decoder_radius = nn.TransformerDecoder(decoder_layers_radius, num_layers)

        self.start_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, z, condition, sequence, pad_mask, tgt_mask):
        batch_size = sequence.size(0)
        start_tokens = self.start_token.expand(batch_size, 1, -1)
        sequence = torch.cat([start_tokens, sequence[:, :-1, :]], dim=1)

        condition_embedding_1 = self.condition_proj_1(condition[:, :3])
        condition_embedding_2 = self.condition_proj_2(condition[:, 3:])

        z_with_condition_1 = torch.cat([z, condition_embedding_1], dim=1)
        z_with_condition_2 = torch.cat([z, condition_embedding_2], dim=1)

        length_logit = self.length_predictor(z_with_condition_1)
        memories_1 = self.latent_to_hidden(z_with_condition_1).unsqueeze(1)
        memories_2 = self.latent_to_hidden(z_with_condition_2).unsqueeze(1)

        sequence = self.input_proj(sequence)
        sequence = self.pos_encoder(sequence)

        condition_expanded_1 = condition_embedding_1.unsqueeze(1).expand(-1, self.max_seq_len, -1)
        condition_expanded_2 = condition_embedding_2.unsqueeze(1).expand(-1, self.max_seq_len, -1)

        sequence_with_condition_1 = torch.cat([sequence, condition_expanded_1], dim=-1)
        output_hidden = self.transformer_decoder(sequence_with_condition_1, memories_1,
                                                 tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask)

        sequence_with_condition_2 = torch.cat([sequence, condition_expanded_2], dim=-1)
        radius_hidden = self.transformer_decoder_radius(sequence_with_condition_2, memories_2,
                                                        tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask)
        output = self.output_proj(output_hidden)
        radius = self.output_proj_radius(radius_hidden)
        output[:, :, 3] = radius.squeeze(-1)
        output = output * (~pad_mask.to(torch.bool)).unsqueeze(-1).float()

        return output, length_logit, radius

    def sample(self, z=None, condition=None, device='cuda'):
        with torch.no_grad():
            if z is None:
                z = torch.randn(1, self.latent_dim, device=device)
            if condition is None:
                condition = torch.randn(1, self.condition_dim, device=device)

            condition_embedding_1 = self.condition_proj_1(condition[:, :3])
            condition_embedding_2 = self.condition_proj_2(condition[:, 3:])

            z_with_condition_1 = torch.cat([z, condition_embedding_1], dim=1)
            z_with_condition_2 = torch.cat([z, condition_embedding_2], dim=1)

            memories_1 = self.latent_to_hidden(z_with_condition_1).unsqueeze(1)
            memories_2 = self.latent_to_hidden(z_with_condition_2).unsqueeze(1)

            len_logit = self.length_predictor(z_with_condition_1)
            length = torch.argmax(len_logit, dim=1).item()
            length = max(length, 5)

            sequence = self.start_token.expand(1, 1, self.input_dim).to(device)
            generated_sequence = []

            for i in range(length):
                tgt = self.input_proj(sequence)
                tgt = self.pos_encoder(tgt)
                condition_expanded_1 = condition_embedding_1.unsqueeze(1).expand(-1, i + 1, -1)
                condition_expanded_2 = condition_embedding_2.unsqueeze(1).expand(-1, i + 1, -1)

                tgt_with_condition_1 = torch.cat([tgt, condition_expanded_1], dim=-1)
                tgt_with_condition_2 = torch.cat([tgt, condition_expanded_2], dim=-1)
                tgt_mask = auto_mask(i + 1).to(device)

                output_1 = self.transformer_decoder(tgt_with_condition_1, memories_1, tgt_mask=tgt_mask)
                output_2 = self.transformer_decoder_radius(tgt_with_condition_2, memories_2, tgt_mask=tgt_mask)

                output = self.output_proj(output_1[:, -1])
                radius = self.output_proj_radius(output_2[:, -1])
                output[:, 3] = radius.squeeze(-1)

                generated_sequence.append(output)
                sequence = torch.cat([sequence, output.unsqueeze(1)], dim=1)

            return torch.cat(generated_sequence, dim=0)


class TransformerVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, nhead, max_seq_len, condition_dim):
        super(TransformerVAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers, nhead, max_seq_len, condition_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, num_layers, nhead, max_seq_len, condition_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, sequence, condition, pad_mask, tgt_mask):
        mu, log_var = self.encoder(sequence, condition, pad_mask)
        z = self.reparameterize(mu, log_var)
        output, length_logit, radius = self.decoder(z, condition, sequence, pad_mask, tgt_mask)
        return output, length_logit, radius, mu, log_var

    def get_loss(self, recon, original, length_logit, mu, log_var, pad_mask, true_radius):
        mask = (~pad_mask.to(torch.bool)).unsqueeze(-1).float()
        recon_loss = F.mse_loss(recon * mask, original * mask, reduction='sum') / torch.sum(mask)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)

        true_length = (~pad_mask.to(torch.bool)).sum(dim=1) - 1
        length_loss = F.cross_entropy(length_logit, true_length)

        predicted_radius = recon[:, :, 3].reshape(recon.shape[0], recon.shape[1])
        radius_loss = F.mse_loss(predicted_radius, true_radius, reduction='mean')

        return recon_loss, kl_loss, length_loss, radius_loss
