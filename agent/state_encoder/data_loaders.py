import uuid, shutil, os
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

import pyarrow.parquet as pq


class ArrowDataLoader:
    def __init__(self, arrow_table, batch_size=2000, default_collate=False):
        self.batch_size = batch_size
        self.arrow_table = arrow_table
        self.default_collate = default_collate
        self._in_iter = None

    def __len__(self):
        return len(self.arrow_table.to_batches(self.batch_size))

    def __iter__(self):
        if not self._in_iter:
            self.arrow_gen = iter(self.arrow_table.to_batches(self.batch_size))
            print('reset iterator to position 0')

        for batch in self.arrow_gen:
            self._in_iter = True
            batch = batch.to_pandas()

            data = {}
            for col in batch.columns:

                col_data = []
                for i, row in batch.iterrows():
                    if isinstance(row[col], dict):
                        if isinstance(row[col]['values'], (np.ndarray, list)):
                            try:
                                t = torch.stack([torch.tensor(d) for d in row[col]['values']])
                            except:
                                t = [torch.tensor(d) for d in row[col]['values']]
                        else:
                            t = torch.tensor([row[col]['values']])
                    else:
                        if isinstance(row[col], (np.ndarray, list)):
                            try:
                                t = torch.stack([torch.tensor(d) for d in row[col]['values']])
                            except:
                                t = [torch.tensor(d) for d in row[col]['values']]
                        else:
                            t = torch.tensor([row[col]])

                    col_data.append(t)

                ragged_check = self.ragged_check(col_data)
                if self.default_collate and len(col_data[0].size()) > 1 and not ragged_check:
                    col_data = pad_sequence(col_data, batch_first=True).float()
                elif ragged_check:
                    col_data = torch.stack(col_data).float()
                else:
                    ### if ragged tensor and no collate then return LIST of ragged tensors instead of TENSOR.
                    pass

                data.update({col: col_data})
            yield data

        self._in_iter = False

    @staticmethod
    def ragged_check(t_list):
        return all(t_list[0].size() == t.size() for t in t_list)


class ArrowDataModule(LightningDataModule):
    def __init__(self, all_data, data_cols, split_on_col='id', split_sizes=[0.8, 0.2], batch_size=1000,
                 temp_storage_path='.', data_id=None, keep_data=False, train_test_split=False):
        super().__init__()
        self.all_data = all_data
        self.split_on_col = split_on_col
        self.split_sizes = split_sizes
        self.batch_size = batch_size
        self.data_cols = data_cols
        self.temp_storage_path = temp_storage_path
        self.data_id = data_id
        self.keep_data = keep_data
        self.train_test_split = train_test_split

    ### called once per training session
    def prepare_data(self):
        if not self.data_id:
            self.data_id = uuid.uuid4()
            print(f'data_id: {self.data_id}')

            if self.train_test_split:
                train_ids, test_ids = self.all_data.select(self.split_on_col).distinct().randomSplit(self.split_sizes, seed=42)
                train_ids, val_ids = train_ids.randomSplit(self.split_sizes, seed=42)

                test_data = self.all_data.join(test_ids, self.split_on_col).select(*self.data_cols)
                val_data = self.all_data.join(val_ids, self.split_on_col).select(*self.data_cols)
                train_data = self.all_data.join(train_ids, self.split_on_col).select(*self.data_cols)

                test_data.write.parquet(os.path.join(self.temp_storage_path, f'test_{self.data_id}'), mode='overwrite')
                val_data.write.parquet(os.path.join(self.temp_storage_path, f'val_{self.data_id}'), mode='overwrite')
                train_data.write.parquet(os.path.join(self.temp_storage_path, f'train_{self.data_id}'), mode='overwrite')

                print('train/test/val splits created')

                print(f'train size: {train_ids.count():,}')
                print(f'test size:  {test_ids.count():,}')
                print(f'val size:   {val_ids.count():,}')

            all_data = self.all_data.select(*self.data_cols)
            all_data.write.parquet(os.path.join(self.temp_storage_path, f'all_{self.data_id}'), mode='overwrite')
            print(f'total size: {self.all_data.select(self.split_on_col).distinct().count():,}')

    # Called once per GPU
    def setup(self, stage=None):
        if self.train_test_split:
            self.train_dataset = pq.read_table(os.path.join(self.temp_storage_path, f'train_{self.data_id}'))
            self.test_dataset = pq.read_table(os.path.join(self.temp_storage_path, f'test_{self.data_id}'))
            self.val_dataset = pq.read_table(os.path.join(self.temp_storage_path, f'val_{self.data_id}'))

        self.all_dataset = pq.read_table(os.path.join(self.temp_storage_path, f'all_{self.data_id}'))

    def train_dataloader(self):
        return ArrowDataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return ArrowDataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return ArrowDataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return ArrowDataLoader(self.all_dataset, batch_size=self.batch_size)

    def teardown(self, stage=None):
        if not self.keep_data:
            if self.train_test_split:
                shutil.rmtree(os.path.join(self.temp_storage_path, f'train_{self.data_id}'))
                shutil.rmtree(os.path.join(self.temp_storage_path, f'test_{self.data_id}'))
                shutil.rmtree(os.path.join(self.temp_storage_path, f'val_{self.data_id}'))
            shutil.rmtree(os.path.join(self.temp_storage_path, f'all_{self.data_id}'))
            print('converters deleted')


class StateDataset(Dataset):
    def __init__(self, data, state_col, max_seq_len=100, pad_token=0, pad_side='right', eos_token=None):
        self.data = data
        self.state_col = state_col
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        self.pad_side = pad_side
        self.eos_token = eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data.iloc[idx][self.state_col]
        X = np.asarray(X).astype('int')
        # if self.eos_token:
        #     X = np.concatenate([np.append(a, self.eos_token) for a in arr])
        # else:
        #     X = np.concatenate(arr)

        if len(X) > self.max_seq_len:
            X = torch.tensor(X[len(X) - self.max_seq_len:len(X)])
        else:
            if self.pad_side == 'right':
                pad = (0, self.max_seq_len - len(X))
            elif self.pad_side == 'left':
                pad = (self.max_seq_len - len(X), 0)
            else:
                raise ValueError('choose left or right for pad_side')

            X = torch.tensor(np.pad(X, pad, mode='constant', constant_values=(self.pad_token, self.pad_token)))

        mask = torch.where(X == self.pad_token, 0, 1)

        return {'obs': X, 'mask': mask}


class StateDataModule(LightningDataModule):
    def __init__(self, all_data, state_col, split_on_col='cagm', split_sizes=[0.8, 0.2], batch_size=1000,
                 samp_size_per_epoch=None, num_workers=6, train_test_split=False, max_seq_len=100,
                 pad_token=0, pad_side='right', eos_token=None):
        super().__init__()
        self.all_data = all_data
        self.state_col = state_col
        self.split_on_col = split_on_col
        self.split_sizes = split_sizes
        self.batch_size = batch_size
        self.samp_size_per_epoch = samp_size_per_epoch
        self.train_test_split = train_test_split
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        self.pad_side = pad_side
        self.eos_token = eos_token

    ### called once per training session
    def prepare_data(self):
        if self.train_test_split:
            keys = self.all_data[[self.split_on_col]].drop_duplicates()
            train, test = train_test_split(keys, train_size=self.split_sizes[0], test_size=self.split_sizes[1],
                                           random_state=42)
            train, val = train_test_split(train, train_size=self.split_sizes[0], test_size=self.split_sizes[1],
                                          random_state=42)

            self.test_data = self.all_data.merge(test, on=self.split_on_col)[[self.state_col]]
            self.val_data = self.all_data.merge(val, on=self.split_on_col)[[self.state_col]]
            self.train_data = self.all_data.merge(train, on=self.split_on_col)[[self.state_col]]

            print('train/test/val splits created')
            print(f'train size: {train.shape[0]:,}')
            print(f'test size:  {test.shape[0]:,}')
            print(f'val size:   {val.shape[0]:,}')

        self.all_data = self.all_data[[self.state_col]]
        print(f'total size: {self.all_data.shape[0]:,}')

    # Called once per GPU
    def setup(self, stage=None):
        if self.train_test_split:
            self.train_dataset = StateDataset(self.train_data, self.state_col, self.max_seq_len, self.pad_token, self.pad_side,
                                            self.eos_token)
            self.test_dataset = StateDataset(self.test_data, self.state_col, self.max_seq_len, self.pad_token, self.pad_side,
                                           self.eos_token)
            self.val_dataset = StateDataset(self.val_data, self.state_col, self.max_seq_len, self.pad_token, self.pad_side,
                                          self.eos_token)
        self.all_dataset = StateDataset(self.all_data, self.state_col, self.max_seq_len, self.pad_token, self.pad_side, self.eos_token)

    def train_dataloader(self):
        if self.samp_size_per_epoch:
            rand_samps = np.random.choice(list(range(len(self.train_dataset))), self.samp_size_per_epoch, replace=False)
            return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers,
                              sampler=SubsetRandomSampler(rand_samps))
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers,
                              shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers,
                          shuffle=False)
