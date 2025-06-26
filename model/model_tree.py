from torch import nn
import torch


class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)
        self.LeakyReLu = nn.LeakyReLU()
        self.latent_dim = feature_size
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        encode = self.LeakyReLu(self.mlp1(input))
        mu, log_var = self.mlp2mu(encode), self.mlp2var(encode)

        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)

        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        z = eps * std + mu
        return torch.cat([z, kl_loss], dim=1)


class InternalEncoder(nn.Module):

    def __init__(self, input_size: int, feature_size: int, hidden_size: int):
        super(InternalEncoder, self).__init__()

        self.attribute_lin_encoder_1 = nn.Linear(input_size, hidden_size)
        self.attribute_lin_encoder_2 = nn.Linear(hidden_size, feature_size)

        self.right_lin_encoder_1 = nn.Linear(feature_size, hidden_size)
        self.right_lin_encoder_2 = nn.Linear(hidden_size, feature_size)

        self.left_lin_encoder_1 = nn.Linear(feature_size, hidden_size)
        self.left_lin_encoder_2 = nn.Linear(hidden_size, feature_size)

        self.final_lin_encoder_1 = nn.Linear(2 * feature_size, feature_size)
        self.LeakyReLu = nn.LeakyReLU()
        self.feature_size = feature_size

    def forward(self, input, right_input, left_input):
        attributes = self.attribute_lin_encoder_1(input)
        attributes = self.LeakyReLu(attributes)
        attributes = self.attribute_lin_encoder_2(attributes)
        attributes = self.LeakyReLu(attributes)

        if right_input is not None:
            context = self.right_lin_encoder_1(right_input)
            context = self.LeakyReLu(context)
            context = self.right_lin_encoder_2(context)
            context = self.LeakyReLu(context)

            if left_input is not None:
                left = self.left_lin_encoder_1(left_input)
                left = self.LeakyReLu(left)
                context += self.left_lin_encoder_2(left)
                context = self.LeakyReLu(context)
        else:
            context = torch.zeros(input.shape[0], self.feature_size, requires_grad=True, device=attributes.device)

        feature = torch.cat((attributes, context), 1)
        feature = self.final_lin_encoder_1(feature)
        feature = self.LeakyReLu(feature)
        return feature


class RecursiveEncoder(nn.Module):

    def __init__(self, input_size: int, feature_size: int, hidden_size: int):
        super(RecursiveEncoder, self).__init__()
        self.leaf_encoder = InternalEncoder(input_size, feature_size, hidden_size)
        self.internal_encoder = InternalEncoder(input_size, feature_size, hidden_size)
        self.bifurcation_encoder = InternalEncoder(input_size, feature_size, hidden_size)
        self.sample_encoder = Sampler(feature_size=feature_size, hidden_size=hidden_size)

    def leafEncoder(self, node, right=None, left=None):
        return self.internal_encoder(node, right, left)

    def internalEncoder(self, node, right, left=None):
        return self.internal_encoder(node, right, left)

    def bifurcationEncoder(self, node, right, left):
        return self.bifurcation_encoder(node, right, left)

    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)


class NodeClassifier(nn.Module):

    def __init__(self, latent_size: int, hidden_size: int):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(latent_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, 3)
        self.LeakyReLu = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.LeakyReLu(output)
        output = self.mlp2(output)
        output = self.LeakyReLu(output)
        output = self.mlp3(output)
        return output


class SampleDecoder(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, feature_size)

        self.LeakyReLu = nn.LeakyReLU()

    def forward(self, input_feature):
        output = self.LeakyReLu(self.mlp1(input_feature))
        output = self.LeakyReLu(self.mlp2(output))
        output = self.mlp3(output)

        return output


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.mlp = nn.Linear(latent_size, hidden_size)
        self.mlp_left = nn.Linear(hidden_size, hidden_size)
        self.mlp_left2 = nn.Linear(hidden_size, latent_size)
        self.mlp_right = nn.Linear(hidden_size, hidden_size)
        self.mlp_right2 = nn.Linear(hidden_size, latent_size)
        self.mlp2 = nn.Linear(hidden_size, latent_size)
        self.mlp3 = nn.Linear(latent_size, output_size)
        self.LeakyReLu = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()
        self.output_size = output_size
        self.mlp4 = nn.Linear(output_size, output_size)

    def common_branch(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.LeakyReLu(vector)
        return vector

    def attr_branch(self, vector):
        vector = self.mlp2(vector)
        vector = self.LeakyReLu(vector)
        vector = self.mlp3(vector)
        # vector = self.LeakyReLu(vector).reshape(-1, 1, self.output_size)
        vector = self.LeakyReLu(vector)
        vector = self.mlp4(vector).reshape(-1, 1, self.output_size)
        sigmoid_part = self.Sigmoid(vector[:, :, 3:7])
        vector = torch.cat([vector[:, :, :3], sigmoid_part, vector[:, :, 7:]], dim=2)

        return vector

    def right_branch(self, vector):
        right_feature = self.mlp_right(vector)
        right_feature = self.LeakyReLu(right_feature)
        right_feature = self.mlp_right2(right_feature)
        right_feature = self.LeakyReLu(right_feature)
        return right_feature

    def left_branch(self, vector):
        left_feature = self.mlp_left(vector)
        left_feature = self.LeakyReLu(left_feature)
        left_feature = self.mlp_left2(left_feature)
        left_feature = self.LeakyReLu(left_feature)
        return left_feature

    def forward(self, parent_feature):
        vector = self.common_branch(parent_feature)
        attr_vector = self.attr_branch(vector)

        return attr_vector

    def forward1(self, parent_feature):
        vector = self.common_branch(parent_feature)
        attr_vector = self.attr_branch(vector)
        right_vector = self.right_branch(vector)
        return right_vector, attr_vector

    def forward2(self, parent_feature):
        vector = self.common_branch(parent_feature)
        attr_vector = self.attr_branch(vector)
        right_vector = self.right_branch(vector)
        left_vector = self.left_branch(vector)
        return left_vector, right_vector, attr_vector


class RecursiveDecoder(nn.Module):

    def __init__(self, latent_size, hidden_size, output_size, args):
        super(RecursiveDecoder, self).__init__()
        self.decoder = Decoder(latent_size, hidden_size, output_size)
        self.node_classifier = NodeClassifier(latent_size, hidden_size)
        self.sample_decoder = SampleDecoder(feature_size=latent_size, hidden_size=hidden_size)
        self.mseLoss = nn.MSELoss()
        self.ceLoss = nn.CrossEntropyLoss()
        self.alfa = 2
        self.device = args.device
        self.input_size = args.input_size

    def featureDecoder(self, feature):
        return self.decoder.forward(feature)

    def internalDecoder(self, feature):
        return self.decoder.forward1(feature)

    def bifurcationDecoder(self, feature):
        return self.decoder.forward2(feature)

    def nodeClassifier(self, feature):
        return self.node_classifier(feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)

    def MSE_loss(self, origin, prediction):
        if origin is None:
            return
        else:
            origin = torch.stack(origin)[:, 0:self.input_size]
            prediction = prediction[:, :, 0:self.input_size]
            loss = [self.mseLoss(pre.reshape(1, self.input_size), gt.reshape(1, self.input_size)) for gt, pre in
                    zip(prediction.reshape(-1, self.input_size), origin.reshape(-1, self.input_size))]

            return loss

    def classifyLossEstimator(self, prediction, label):
        if label is None:
            return None
        vector_map = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=self.device, dtype=torch.float)
        label = vector_map[label]
        prediction = prediction.reshape(-1, 3)
        loss = [self.ceLoss(pre.unsqueeze(0), gt.unsqueeze(0)) * self.alfa for pre, gt in zip(prediction, label)]
        return loss

    def Add(self, vector1, vector2):
        return vector1 + vector2

    def Mult(self, weight, vector):
        weight = torch.tensor(weight, device=self.device)
        return weight * vector
