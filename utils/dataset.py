import os

import torch
from torch.utils.data import Dataset


##############################################################
# Sequence Dataset#
##############################################################

class Skeleton(Dataset):
    def __init__(self, data_path, dataset_name, max_seq_len, is_train=True):
        self.training = True
        self.data_path = data_path
        self.dataset = dataset_name
        self.max_seq_len = max_seq_len
        self.save_path = os.path.join(data_path, '{}.pt'.format(dataset_name))

        data = torch.load(self.save_path)
        self.data = data['data']
        self.length = data['length']
        self.condition = data['condition']

        split_idx = int(len(self.data) * 0.8)
        self.train_data = self.data[:split_idx]
        self.test_data = self.data[split_idx:]
        self.train_condition = self.condition[:split_idx]
        self.test_condition = self.condition[split_idx:]

        if is_train is True:
            self.training = True
        else:
            self.training = False

    def __len__(self):
        if self.training:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.training:
            return self.train_data[idx], self.train_condition[idx]
        else:
            return self.test_data[idx], self.test_condition[idx]


##############################################################
# Tree Dataset #
##############################################################

def coll_function(batch):
    trees, num_nodes, graphs, file_names = zip(*batch)
    return trees, num_nodes, graphs, file_names


class TreeDataset(Dataset):
    def __init__(self, dataset_name, data_dir, is_train=True):
        if is_train is True:
            self.training = True
        else:
            self.training = False
        self.pt_file = os.path.join(data_dir, f"{dataset_name}.pt")

        self.data = []
        self.graphs = []
        self.file_names = []
        self.num_nodes = []
        self.max_nodes = 0

        if os.path.exists(self.pt_file):
            self.load_data()
        else:
            pass

        split_idx = int(len(self.data) * 0.9)
        self.train_data = self.data[:split_idx]
        self.test_data = self.data[split_idx:]
        self.train_graphs = self.graphs[:split_idx]
        self.test_graphs = self.graphs[split_idx:]

        self.train_file_names = self.file_names[:split_idx]
        self.test_file_names = self.file_names[split_idx:]
        self.train_num_nodes = self.num_nodes[:split_idx]
        self.test_num_nodes = self.num_nodes[split_idx:]

    def __len__(self):
        if self.training is True:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.training is True:
            num_nodes = torch.tensor(self.train_num_nodes[idx])
            return self.train_data[idx], num_nodes, self.train_graphs[idx], self.train_file_names[idx]
        else:
            num_nodes = torch.tensor(self.test_num_nodes[idx])
            return self.test_data[idx], num_nodes, self.test_graphs[idx], self.test_file_names[idx]

    def load_data(self):
        loaded_data = torch.load(self.pt_file)
        self.data = loaded_data['data']
        self.graphs = loaded_data['graphs']
        self.num_nodes = loaded_data['num_nodes']
        self.file_names = loaded_data['file_names']
