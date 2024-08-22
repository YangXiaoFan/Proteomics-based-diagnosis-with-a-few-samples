# coding: UTF-8
import torch
import numpy as np
import random
import time
from datetime import timedelta
import pandas as pd


def load_dataset(path):
    set_1 = pd.read_excel(path, sheet_name = 0)
    set_2 = pd.read_excel(path, sheet_name = 1)

    label_1 = set_1.iloc[0].tolist()
    label_2 = set_2.iloc[0].tolist()
    expression_1 = set_1.iloc[1:5688].values
    expression_2 = set_2.iloc[1:5688].values

    dataset_1 = []
    dataset_2 = []

    for col in range(expression_1.shape[1]):
        expression = expression_1[:, col]
        expression = np.append(expression, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
        label = label_1[col]
        label = 1 if label == "T" else 0
        expression_pad = (expression != 0).astype(int)
        dataset_1.append((expression, label, expression_pad))

    for col in range(expression_2.shape[1]):
        expression = expression_2[:, col]
        expression = np.append(expression, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
        label = label_2[col]
        label = 1 if label == "T" else 0
        expression_pad = (expression != 0).astype(int)
        dataset_2.append((expression, label, expression_pad))    
    
    set_1_order = np.array(range(0, len(dataset_1) // 2), dtype=int)
    random.shuffle(set_1_order)

    set_2_order = np.array(range(0, len(dataset_2) // 2), dtype=int)
    random.shuffle(set_2_order)

    set_1_train = []
    set_1_dev = [] 
    set_2_train = [] 
    set_2_test = []

    for i in range(0, 87):
        set_1_train.append(dataset_1[set_1_order[i] * 2])
        set_1_train.append(dataset_1[set_1_order[i] * 2 + 1])

    for i in range(87, 124):
        set_1_dev.append(dataset_1[set_1_order[i] * 2])
        set_1_dev.append(dataset_1[set_1_order[i] * 2 + 1])

    for i in range(0, 6):
        set_2_train.append(dataset_2[set_2_order[i] * 2])
        set_2_train.append(dataset_2[set_2_order[i] * 2 + 1])

    for i in range(6, 60):
        set_2_test.append(dataset_2[set_2_order[i] * 2])
        set_2_test.append(dataset_2[set_2_order[i] * 2 + 1])

    return set_1_train, set_1_dev, set_2_train, set_2_test
            # [([...], 0, [...]), ([...], 1, [...]), ...]
            # 数据集1的70%作为训练集，30%作为验证集
            # 数据集2的10%作为训练集，90%作为测试集


def build_dataset(config):
    set_1_train, set_1_dev, set_2_train, set_2_test = load_dataset(config.data_path)
    return set_1_train, set_1_dev, set_2_train, set_2_test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, aug):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.aug = aug

    def _to_tensor(self, datas):
        x = np.array([_[0].astype(np.float64) for _ in datas])
        x = torch.FloatTensor(x).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        pad = np.array([_[2].astype(np.float64) for _ in datas])
        pad = torch.FloatTensor(pad).to(self.device)

        if self.aug == True:
            x_aug = x
            pad_aug = pad
            random_tensor = torch.rand(x.shape)
            x_aug[random_tensor < 0.5] = 0
            pad_aug[random_tensor < 0.5] = 0
            return (x_aug, pad_aug), (y, x, pad)
        else:
            return (x, pad), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, aug = False):
    iter = DatasetIterater(dataset, config.batch_size, config.device, aug)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
