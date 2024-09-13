import os
import torch
import numpy as np
import torch.utils.data as data
import random
import pytorch_lightning as pl

from src.utils.data_utils import crop_sequence


class DATA_proc(data.Dataset):
    def __init__(self, dataname, train, Ttot, random_crop, inputpc_size, sigma, add_random_noise, seed):
        self.split = 'train' if train else 'test'
        self.dataname = dataname
        self.root = os.path.join('data', self.dataname, self.split)
        self.T = Ttot
        self.random_crop = random_crop
        self.inputpc_size = inputpc_size
        self.sigma = sigma
        self.add_random_noise = add_random_noise

        data_list = sorted(os.listdir(self.root))
        self.seq_path = []
        for dataset in data_list:
            sid_list = sorted(os.listdir(os.path.join(self.root, dataset)))
            for sid in sid_list:
                seq_list = sorted(os.listdir(os.path.join(self.root, dataset, sid)))
                for seq in seq_list:
                    self.seq_path.append(os.path.join(self.root, dataset, sid, seq))

        random.seed(seed)
        # random.shuffle(self.seq_path)

    def __getitem__(self, index):
        npz_data = np.load(self.seq_path[index])

        x = npz_data['points']
        joints = npz_data['joints']

        if self.dataname.startswith('human'):
            body_pose = npz_data['body_pose'][:, :66]
        elif self.dataname.startswith('hand'):
            body_pose = npz_data['body_pose'][:, :48]
        body_shape = npz_data['body_shape'][:, :10]
        trans = npz_data['trans']

        partial_pcd = 0
        
        time_length = len(trans)

        if self.random_crop:
            rand_start = time_length - 1 - (self.T - 1)
            if rand_start < 0:
                start = 0
            else:
                start = random.randint(0, time_length - 1 - (self.T - 1))
            # start = 0
        else:
            offset = (self.epoch_id % self.T)
            start = self.epoch_id % (time_length // (self.T)) * (self.T) + offset
            if start + (self.T - 1) >= x.shape[0]:
                start = max(start - 2 * offset, 0)

        x = crop_sequence(x, start, self.T)
        body_pose = crop_sequence(body_pose, start, self.T)
        body_shape = crop_sequence(body_shape, start, self.T)
        trans = crop_sequence(trans, start, self.T)
        joints = crop_sequence(joints, start, self.T)

        sample_idx = np.random.permutation(x.shape[1])[:self.inputpc_size]
        x = x[:, sample_idx]
        # trans[0] = [0.0, 0.2, 0.9]
        if self.dataname.endswith('partial'):
            partial_pcd = npz_data['partial_points']
            partial_pcd = crop_sequence(partial_pcd, start, self.T)
            sample_idx_p = np.random.permutation(partial_pcd.shape[1])[:self.inputpc_size]
            partial_pcd = partial_pcd[:, sample_idx_p]
            if self.dataname.startswith('human'):
                partial_pcd -= trans[0]
            if self.sigma != 0:
                noise = np.random.normal(0.0, self.sigma, partial_pcd.shape)
                partial_pcd = partial_pcd + noise

        if self.dataname.startswith('human'):
            x -= trans[0]
            joints -= trans[0]
            trans = trans - trans[0]

        if self.dataname.endswith('partial'):
            return index, x, joints, body_pose, body_shape, trans, partial_pcd
        else:
            return index, x, joints, body_pose, body_shape, trans, x

    def __len__(self):
        if self.split == 'train':
            return len(self.seq_path) // 10
        else:
            return len(self.seq_path)


class DATA_PL(pl.LightningDataModule):
    def __init__(self, options):
        super().__init__()
        self.batch_size = options.nbatch

        self.dataset = options.dataset
        self.T = options.Ttot
        self.random_crop = bool(options.random_crop)
        self.inputpc_size = options.inputpc_size
        self.seed = options.seed
        self.num_workers = options.num_workers
        self.sigma = options.noise_sigma
        self.add_random_noise = options.add_random_noise

    def setup(self, stage=None):
        self.train_data = DATA_proc(self.dataset, True, self.T, self.random_crop, self.inputpc_size, self.sigma, self.add_random_noise, self.seed)
        self.val_data = DATA_proc(self.dataset, False, self.T, self.random_crop, self.inputpc_size, self.sigma, self.add_random_noise, self.seed)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           shuffle=False,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)

